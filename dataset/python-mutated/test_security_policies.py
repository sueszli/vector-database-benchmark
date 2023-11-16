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
from google.cloud.compute_v1.services.security_policies import SecurityPoliciesClient, pagers, transports
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
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert SecurityPoliciesClient._get_default_mtls_endpoint(None) is None
    assert SecurityPoliciesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert SecurityPoliciesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert SecurityPoliciesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert SecurityPoliciesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert SecurityPoliciesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(SecurityPoliciesClient, 'rest')])
def test_security_policies_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.SecurityPoliciesRestTransport, 'rest')])
def test_security_policies_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(SecurityPoliciesClient, 'rest')])
def test_security_policies_client_from_service_account_file(client_class, transport_name):
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

def test_security_policies_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = SecurityPoliciesClient.get_transport_class()
    available_transports = [transports.SecurityPoliciesRestTransport]
    assert transport in available_transports
    transport = SecurityPoliciesClient.get_transport_class('rest')
    assert transport == transports.SecurityPoliciesRestTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SecurityPoliciesClient, transports.SecurityPoliciesRestTransport, 'rest')])
@mock.patch.object(SecurityPoliciesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecurityPoliciesClient))
def test_security_policies_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(SecurityPoliciesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(SecurityPoliciesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(SecurityPoliciesClient, transports.SecurityPoliciesRestTransport, 'rest', 'true'), (SecurityPoliciesClient, transports.SecurityPoliciesRestTransport, 'rest', 'false')])
@mock.patch.object(SecurityPoliciesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecurityPoliciesClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_security_policies_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        return 10
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

@pytest.mark.parametrize('client_class', [SecurityPoliciesClient])
@mock.patch.object(SecurityPoliciesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SecurityPoliciesClient))
def test_security_policies_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SecurityPoliciesClient, transports.SecurityPoliciesRestTransport, 'rest')])
def test_security_policies_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SecurityPoliciesClient, transports.SecurityPoliciesRestTransport, 'rest', None)])
def test_security_policies_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('request_type', [compute.AddRuleSecurityPolicyRequest, dict])
def test_add_rule_rest(request_type):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request_init['security_policy_rule_resource'] = {'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}
    test_field = compute.AddRuleSecurityPolicyRequest.meta.fields['security_policy_rule_resource']

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
    for (field, value) in request_init['security_policy_rule_resource'].items():
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
                for i in range(0, len(request_init['security_policy_rule_resource'][field])):
                    del request_init['security_policy_rule_resource'][field][i][subfield]
            else:
                del request_init['security_policy_rule_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_rule(request)
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

def test_add_rule_rest_required_fields(request_type=compute.AddRuleSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_rule._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_rule._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_rule(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_rule_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_rule._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly',)) & set(('project', 'securityPolicy', 'securityPolicyRuleResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_rule_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_add_rule') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_add_rule') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddRuleSecurityPolicyRequest.pb(compute.AddRuleSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddRuleSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_rule(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_rule_rest_bad_request(transport: str='rest', request_type=compute.AddRuleSecurityPolicyRequest):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_rule(request)

def test_add_rule_rest_flattened():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_rule(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}/addRule' % client.transport._host, args[1])

def test_add_rule_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_rule(compute.AddRuleSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))

def test_add_rule_rest_error():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AddRuleSecurityPolicyRequest, dict])
def test_add_rule_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request_init['security_policy_rule_resource'] = {'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}
    test_field = compute.AddRuleSecurityPolicyRequest.meta.fields['security_policy_rule_resource']

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
    for (field, value) in request_init['security_policy_rule_resource'].items():
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
                for i in range(0, len(request_init['security_policy_rule_resource'][field])):
                    del request_init['security_policy_rule_resource'][field][i][subfield]
            else:
                del request_init['security_policy_rule_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_rule_unary(request)
    assert isinstance(response, compute.Operation)

def test_add_rule_unary_rest_required_fields(request_type=compute.AddRuleSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_rule._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_rule._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_rule_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_rule_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_rule._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly',)) & set(('project', 'securityPolicy', 'securityPolicyRuleResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_rule_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_add_rule') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_add_rule') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddRuleSecurityPolicyRequest.pb(compute.AddRuleSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddRuleSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_rule_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_rule_unary_rest_bad_request(transport: str='rest', request_type=compute.AddRuleSecurityPolicyRequest):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_rule_unary(request)

def test_add_rule_unary_rest_flattened():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_rule_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}/addRule' % client.transport._host, args[1])

def test_add_rule_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_rule_unary(compute.AddRuleSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))

def test_add_rule_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AggregatedListSecurityPoliciesRequest, dict])
def test_aggregated_list_rest(request_type):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPoliciesAggregatedList(etag='etag_value', id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value', unreachables=['unreachables_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPoliciesAggregatedList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.aggregated_list(request)
    assert isinstance(response, pagers.AggregatedListPager)
    assert response.etag == 'etag_value'
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'
    assert response.unreachables == ['unreachables_value']

def test_aggregated_list_rest_required_fields(request_type=compute.AggregatedListSecurityPoliciesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecurityPoliciesRestTransport
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
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.SecurityPoliciesAggregatedList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.SecurityPoliciesAggregatedList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.aggregated_list(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_aggregated_list_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.aggregated_list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_aggregated_list_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_aggregated_list') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_aggregated_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AggregatedListSecurityPoliciesRequest.pb(compute.AggregatedListSecurityPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.SecurityPoliciesAggregatedList.to_json(compute.SecurityPoliciesAggregatedList())
        request = compute.AggregatedListSecurityPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.SecurityPoliciesAggregatedList()
        client.aggregated_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_aggregated_list_rest_bad_request(transport: str='rest', request_type=compute.AggregatedListSecurityPoliciesRequest):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPoliciesAggregatedList()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPoliciesAggregatedList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.aggregated_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/aggregated/securityPolicies' % client.transport._host, args[1])

def test_aggregated_list_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.aggregated_list(compute.AggregatedListSecurityPoliciesRequest(), project='project_value')

def test_aggregated_list_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.SecurityPoliciesAggregatedList(items={'a': compute.SecurityPoliciesScopedList(), 'b': compute.SecurityPoliciesScopedList(), 'c': compute.SecurityPoliciesScopedList()}, next_page_token='abc'), compute.SecurityPoliciesAggregatedList(items={}, next_page_token='def'), compute.SecurityPoliciesAggregatedList(items={'g': compute.SecurityPoliciesScopedList()}, next_page_token='ghi'), compute.SecurityPoliciesAggregatedList(items={'h': compute.SecurityPoliciesScopedList(), 'i': compute.SecurityPoliciesScopedList()}))
        response = response + response
        response = tuple((compute.SecurityPoliciesAggregatedList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1'}
        pager = client.aggregated_list(request=sample_request)
        assert isinstance(pager.get('a'), compute.SecurityPoliciesScopedList)
        assert pager.get('h') is None
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tuple) for i in results))
        for result in results:
            assert isinstance(result, tuple)
            assert tuple((type(t) for t in result)) == (str, compute.SecurityPoliciesScopedList)
        assert pager.get('a') is None
        assert isinstance(pager.get('h'), compute.SecurityPoliciesScopedList)
        pages = list(client.aggregated_list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.DeleteSecurityPolicyRequest, dict])
def test_delete_rest(request_type):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
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

def test_delete_rest_required_fields(request_type=compute.DeleteSecurityPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'securityPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteSecurityPolicyRequest.pb(compute.DeleteSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_rest_bad_request(transport: str='rest', request_type=compute.DeleteSecurityPolicyRequest):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
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
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}' % client.transport._host, args[1])

def test_delete_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete(compute.DeleteSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value')

def test_delete_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteSecurityPolicyRequest, dict])
def test_delete_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
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

def test_delete_unary_rest_required_fields(request_type=compute.DeleteSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'securityPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteSecurityPolicyRequest.pb(compute.DeleteSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteSecurityPolicyRequest):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
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
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}' % client.transport._host, args[1])

def test_delete_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_unary(compute.DeleteSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value')

def test_delete_unary_rest_error():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetSecurityPolicyRequest, dict])
def test_get_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPolicy(creation_timestamp='creation_timestamp_value', description='description_value', fingerprint='fingerprint_value', id=205, kind='kind_value', label_fingerprint='label_fingerprint_value', name='name_value', region='region_value', self_link='self_link_value', type_='type__value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get(request)
    assert isinstance(response, compute.SecurityPolicy)
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.fingerprint == 'fingerprint_value'
    assert response.id == 205
    assert response.kind == 'kind_value'
    assert response.label_fingerprint == 'label_fingerprint_value'
    assert response.name == 'name_value'
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.type_ == 'type__value'

def test_get_rest_required_fields(request_type=compute.GetSecurityPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.SecurityPolicy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.SecurityPolicy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('project', 'securityPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_get') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_get') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetSecurityPolicyRequest.pb(compute.GetSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.SecurityPolicy.to_json(compute.SecurityPolicy())
        request = compute.GetSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.SecurityPolicy()
        client.get(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rest_bad_request(transport: str='rest', request_type=compute.GetSecurityPolicyRequest):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get(request)

def test_get_rest_flattened():
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPolicy()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPolicy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}' % client.transport._host, args[1])

def test_get_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get(compute.GetSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value')

def test_get_rest_error():
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetRuleSecurityPolicyRequest, dict])
def test_get_rule_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPolicyRule(action='action_value', description='description_value', kind='kind_value', preview=True, priority=898)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPolicyRule.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_rule(request)
    assert isinstance(response, compute.SecurityPolicyRule)
    assert response.action == 'action_value'
    assert response.description == 'description_value'
    assert response.kind == 'kind_value'
    assert response.preview is True
    assert response.priority == 898

def test_get_rule_rest_required_fields(request_type=compute.GetRuleSecurityPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_rule._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_rule._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('priority',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.SecurityPolicyRule()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.SecurityPolicyRule.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_rule(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_rule_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_rule._get_unset_required_fields({})
    assert set(unset_fields) == set(('priority',)) & set(('project', 'securityPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rule_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_get_rule') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_get_rule') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetRuleSecurityPolicyRequest.pb(compute.GetRuleSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.SecurityPolicyRule.to_json(compute.SecurityPolicyRule())
        request = compute.GetRuleSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.SecurityPolicyRule()
        client.get_rule(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rule_rest_bad_request(transport: str='rest', request_type=compute.GetRuleSecurityPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_rule(request)

def test_get_rule_rest_flattened():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPolicyRule()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPolicyRule.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_rule(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}/getRule' % client.transport._host, args[1])

def test_get_rule_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_rule(compute.GetRuleSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value')

def test_get_rule_rest_error():
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertSecurityPolicyRequest, dict])
def test_insert_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request_init['security_policy_resource'] = {'adaptive_protection_config': {'layer7_ddos_defense_config': {'enable': True, 'rule_visibility': 'rule_visibility_value'}}, 'advanced_options_config': {'json_custom_config': {'content_types': ['content_types_value1', 'content_types_value2']}, 'json_parsing': 'json_parsing_value', 'log_level': 'log_level_value'}, 'creation_timestamp': 'creation_timestamp_value', 'ddos_protection_config': {'ddos_protection': 'ddos_protection_value'}, 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'name': 'name_value', 'recaptcha_options_config': {'redirect_site_key': 'redirect_site_key_value'}, 'region': 'region_value', 'rules': [{'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}], 'self_link': 'self_link_value', 'type_': 'type__value'}
    test_field = compute.InsertSecurityPolicyRequest.meta.fields['security_policy_resource']

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
    for (field, value) in request_init['security_policy_resource'].items():
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
                for i in range(0, len(request_init['security_policy_resource'][field])):
                    del request_init['security_policy_resource'][field][i][subfield]
            else:
                del request_init['security_policy_resource'][field][subfield]
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

def test_insert_rest_required_fields(request_type=compute.InsertSecurityPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'validateOnly')) & set(('project', 'securityPolicyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertSecurityPolicyRequest.pb(compute.InsertSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_rest_bad_request(transport: str='rest', request_type=compute.InsertSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies' % client.transport._host, args[1])

def test_insert_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert(compute.InsertSecurityPolicyRequest(), project='project_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))

def test_insert_rest_error():
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertSecurityPolicyRequest, dict])
def test_insert_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request_init['security_policy_resource'] = {'adaptive_protection_config': {'layer7_ddos_defense_config': {'enable': True, 'rule_visibility': 'rule_visibility_value'}}, 'advanced_options_config': {'json_custom_config': {'content_types': ['content_types_value1', 'content_types_value2']}, 'json_parsing': 'json_parsing_value', 'log_level': 'log_level_value'}, 'creation_timestamp': 'creation_timestamp_value', 'ddos_protection_config': {'ddos_protection': 'ddos_protection_value'}, 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'name': 'name_value', 'recaptcha_options_config': {'redirect_site_key': 'redirect_site_key_value'}, 'region': 'region_value', 'rules': [{'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}], 'self_link': 'self_link_value', 'type_': 'type__value'}
    test_field = compute.InsertSecurityPolicyRequest.meta.fields['security_policy_resource']

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
    for (field, value) in request_init['security_policy_resource'].items():
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
                for i in range(0, len(request_init['security_policy_resource'][field])):
                    del request_init['security_policy_resource'][field][i][subfield]
            else:
                del request_init['security_policy_resource'][field][subfield]
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

def test_insert_unary_rest_required_fields(request_type=compute.InsertSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'validateOnly')) & set(('project', 'securityPolicyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertSecurityPolicyRequest.pb(compute.InsertSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.InsertSecurityPolicyRequest):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies' % client.transport._host, args[1])

def test_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert_unary(compute.InsertSecurityPolicyRequest(), project='project_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))

def test_insert_unary_rest_error():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ListSecurityPoliciesRequest, dict])
def test_list_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPolicyList(id='id_value', kind='kind_value', next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPolicyList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list(request)
    assert isinstance(response, pagers.ListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'

def test_list_rest_required_fields(request_type=compute.ListSecurityPoliciesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecurityPoliciesRestTransport
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
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.SecurityPolicyList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.SecurityPolicyList.pb(return_value)
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
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_list') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListSecurityPoliciesRequest.pb(compute.ListSecurityPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.SecurityPolicyList.to_json(compute.SecurityPolicyList())
        request = compute.ListSecurityPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.SecurityPolicyList()
        client.list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rest_bad_request(transport: str='rest', request_type=compute.ListSecurityPoliciesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPolicyList()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPolicyList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies' % client.transport._host, args[1])

def test_list_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list(compute.ListSecurityPoliciesRequest(), project='project_value')

def test_list_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.SecurityPolicyList(items=[compute.SecurityPolicy(), compute.SecurityPolicy(), compute.SecurityPolicy()], next_page_token='abc'), compute.SecurityPolicyList(items=[], next_page_token='def'), compute.SecurityPolicyList(items=[compute.SecurityPolicy()], next_page_token='ghi'), compute.SecurityPolicyList(items=[compute.SecurityPolicy(), compute.SecurityPolicy()]))
        response = response + response
        response = tuple((compute.SecurityPolicyList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1'}
        pager = client.list(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.SecurityPolicy) for i in results))
        pages = list(client.list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.ListPreconfiguredExpressionSetsSecurityPoliciesRequest, dict])
def test_list_preconfigured_expression_sets_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_preconfigured_expression_sets(request)
    assert isinstance(response, compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse)

def test_list_preconfigured_expression_sets_rest_required_fields(request_type=compute.ListPreconfiguredExpressionSetsSecurityPoliciesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_preconfigured_expression_sets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_preconfigured_expression_sets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_preconfigured_expression_sets(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_preconfigured_expression_sets_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_preconfigured_expression_sets._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_preconfigured_expression_sets_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_list_preconfigured_expression_sets') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_list_preconfigured_expression_sets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListPreconfiguredExpressionSetsSecurityPoliciesRequest.pb(compute.ListPreconfiguredExpressionSetsSecurityPoliciesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse.to_json(compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse())
        request = compute.ListPreconfiguredExpressionSetsSecurityPoliciesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse()
        client.list_preconfigured_expression_sets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_preconfigured_expression_sets_rest_bad_request(transport: str='rest', request_type=compute.ListPreconfiguredExpressionSetsSecurityPoliciesRequest):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_preconfigured_expression_sets(request)

def test_list_preconfigured_expression_sets_rest_flattened():
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.SecurityPoliciesListPreconfiguredExpressionSetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_preconfigured_expression_sets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/listPreconfiguredExpressionSets' % client.transport._host, args[1])

def test_list_preconfigured_expression_sets_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_preconfigured_expression_sets(compute.ListPreconfiguredExpressionSetsSecurityPoliciesRequest(), project='project_value')

def test_list_preconfigured_expression_sets_rest_error():
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchSecurityPolicyRequest, dict])
def test_patch_rest(request_type):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request_init['security_policy_resource'] = {'adaptive_protection_config': {'layer7_ddos_defense_config': {'enable': True, 'rule_visibility': 'rule_visibility_value'}}, 'advanced_options_config': {'json_custom_config': {'content_types': ['content_types_value1', 'content_types_value2']}, 'json_parsing': 'json_parsing_value', 'log_level': 'log_level_value'}, 'creation_timestamp': 'creation_timestamp_value', 'ddos_protection_config': {'ddos_protection': 'ddos_protection_value'}, 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'name': 'name_value', 'recaptcha_options_config': {'redirect_site_key': 'redirect_site_key_value'}, 'region': 'region_value', 'rules': [{'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}], 'self_link': 'self_link_value', 'type_': 'type__value'}
    test_field = compute.PatchSecurityPolicyRequest.meta.fields['security_policy_resource']

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
    for (field, value) in request_init['security_policy_resource'].items():
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
                for i in range(0, len(request_init['security_policy_resource'][field])):
                    del request_init['security_policy_resource'][field][i][subfield]
            else:
                del request_init['security_policy_resource'][field][subfield]
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

def test_patch_rest_required_fields(request_type=compute.PatchSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'securityPolicy', 'securityPolicyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_patch') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchSecurityPolicyRequest.pb(compute.PatchSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_rest_bad_request(transport: str='rest', request_type=compute.PatchSecurityPolicyRequest):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch(request)

def test_patch_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}' % client.transport._host, args[1])

def test_patch_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch(compute.PatchSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))

def test_patch_rest_error():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchSecurityPolicyRequest, dict])
def test_patch_unary_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request_init['security_policy_resource'] = {'adaptive_protection_config': {'layer7_ddos_defense_config': {'enable': True, 'rule_visibility': 'rule_visibility_value'}}, 'advanced_options_config': {'json_custom_config': {'content_types': ['content_types_value1', 'content_types_value2']}, 'json_parsing': 'json_parsing_value', 'log_level': 'log_level_value'}, 'creation_timestamp': 'creation_timestamp_value', 'ddos_protection_config': {'ddos_protection': 'ddos_protection_value'}, 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'name': 'name_value', 'recaptcha_options_config': {'redirect_site_key': 'redirect_site_key_value'}, 'region': 'region_value', 'rules': [{'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}], 'self_link': 'self_link_value', 'type_': 'type__value'}
    test_field = compute.PatchSecurityPolicyRequest.meta.fields['security_policy_resource']

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
    for (field, value) in request_init['security_policy_resource'].items():
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
                for i in range(0, len(request_init['security_policy_resource'][field])):
                    del request_init['security_policy_resource'][field][i][subfield]
            else:
                del request_init['security_policy_resource'][field][subfield]
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

def test_patch_unary_rest_required_fields(request_type=compute.PatchSecurityPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'securityPolicy', 'securityPolicyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_patch') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchSecurityPolicyRequest.pb(compute.PatchSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_unary_rest_bad_request(transport: str='rest', request_type=compute.PatchSecurityPolicyRequest):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_unary(request)

def test_patch_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}' % client.transport._host, args[1])

def test_patch_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_unary(compute.PatchSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value', security_policy_resource=compute.SecurityPolicy(adaptive_protection_config=compute.SecurityPolicyAdaptiveProtectionConfig(layer7_ddos_defense_config=compute.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(enable=True))))

def test_patch_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchRuleSecurityPolicyRequest, dict])
def test_patch_rule_rest(request_type):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request_init['security_policy_rule_resource'] = {'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}
    test_field = compute.PatchRuleSecurityPolicyRequest.meta.fields['security_policy_rule_resource']

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
    for (field, value) in request_init['security_policy_rule_resource'].items():
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
                for i in range(0, len(request_init['security_policy_rule_resource'][field])):
                    del request_init['security_policy_rule_resource'][field][i][subfield]
            else:
                del request_init['security_policy_rule_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.patch_rule(request)
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

def test_patch_rule_rest_required_fields(request_type=compute.PatchRuleSecurityPolicyRequest):
    if False:
        return 10
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_rule._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_rule._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('priority', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.patch_rule(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_patch_rule_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch_rule._get_unset_required_fields({})
    assert set(unset_fields) == set(('priority', 'validateOnly')) & set(('project', 'securityPolicy', 'securityPolicyRuleResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_rule_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_patch_rule') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_patch_rule') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchRuleSecurityPolicyRequest.pb(compute.PatchRuleSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchRuleSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_rule(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_rule_rest_bad_request(transport: str='rest', request_type=compute.PatchRuleSecurityPolicyRequest):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_rule(request)

def test_patch_rule_rest_flattened():
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.patch_rule(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}/patchRule' % client.transport._host, args[1])

def test_patch_rule_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_rule(compute.PatchRuleSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))

def test_patch_rule_rest_error():
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchRuleSecurityPolicyRequest, dict])
def test_patch_rule_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request_init['security_policy_rule_resource'] = {'action': 'action_value', 'description': 'description_value', 'header_action': {'request_headers_to_adds': [{'header_name': 'header_name_value', 'header_value': 'header_value_value'}]}, 'kind': 'kind_value', 'match': {'config': {'src_ip_ranges': ['src_ip_ranges_value1', 'src_ip_ranges_value2']}, 'expr': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'versioned_expr': 'versioned_expr_value'}, 'preconfigured_waf_config': {'exclusions': [{'request_cookies_to_exclude': [{'op': 'op_value', 'val': 'val_value'}], 'request_headers_to_exclude': {}, 'request_query_params_to_exclude': {}, 'request_uris_to_exclude': {}, 'target_rule_ids': ['target_rule_ids_value1', 'target_rule_ids_value2'], 'target_rule_set': 'target_rule_set_value'}]}, 'preview': True, 'priority': 898, 'rate_limit_options': {'ban_duration_sec': 1680, 'ban_threshold': {'count': 553, 'interval_sec': 1279}, 'conform_action': 'conform_action_value', 'enforce_on_key': 'enforce_on_key_value', 'enforce_on_key_configs': [{'enforce_on_key_name': 'enforce_on_key_name_value', 'enforce_on_key_type': 'enforce_on_key_type_value'}], 'enforce_on_key_name': 'enforce_on_key_name_value', 'exceed_action': 'exceed_action_value', 'exceed_redirect_options': {'target': 'target_value', 'type_': 'type__value'}, 'rate_limit_threshold': {}}, 'redirect_options': {}}
    test_field = compute.PatchRuleSecurityPolicyRequest.meta.fields['security_policy_rule_resource']

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
    for (field, value) in request_init['security_policy_rule_resource'].items():
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
                for i in range(0, len(request_init['security_policy_rule_resource'][field])):
                    del request_init['security_policy_rule_resource'][field][i][subfield]
            else:
                del request_init['security_policy_rule_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.patch_rule_unary(request)
    assert isinstance(response, compute.Operation)

def test_patch_rule_unary_rest_required_fields(request_type=compute.PatchRuleSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_rule._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_rule._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('priority', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.patch_rule_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_patch_rule_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch_rule._get_unset_required_fields({})
    assert set(unset_fields) == set(('priority', 'validateOnly')) & set(('project', 'securityPolicy', 'securityPolicyRuleResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_rule_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_patch_rule') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_patch_rule') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchRuleSecurityPolicyRequest.pb(compute.PatchRuleSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchRuleSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_rule_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_rule_unary_rest_bad_request(transport: str='rest', request_type=compute.PatchRuleSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_rule_unary(request)

def test_patch_rule_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.patch_rule_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}/patchRule' % client.transport._host, args[1])

def test_patch_rule_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_rule_unary(compute.PatchRuleSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value', security_policy_rule_resource=compute.SecurityPolicyRule(action='action_value'))

def test_patch_rule_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.RemoveRuleSecurityPolicyRequest, dict])
def test_remove_rule_rest(request_type):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_rule(request)
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

def test_remove_rule_rest_required_fields(request_type=compute.RemoveRuleSecurityPolicyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_rule._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_rule._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('priority',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.remove_rule(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_rule_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_rule._get_unset_required_fields({})
    assert set(unset_fields) == set(('priority',)) & set(('project', 'securityPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_rule_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_remove_rule') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_remove_rule') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.RemoveRuleSecurityPolicyRequest.pb(compute.RemoveRuleSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.RemoveRuleSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.remove_rule(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_rule_rest_bad_request(transport: str='rest', request_type=compute.RemoveRuleSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_rule(request)

def test_remove_rule_rest_flattened():
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.remove_rule(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}/removeRule' % client.transport._host, args[1])

def test_remove_rule_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.remove_rule(compute.RemoveRuleSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value')

def test_remove_rule_rest_error():
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.RemoveRuleSecurityPolicyRequest, dict])
def test_remove_rule_unary_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_rule_unary(request)
    assert isinstance(response, compute.Operation)

def test_remove_rule_unary_rest_required_fields(request_type=compute.RemoveRuleSecurityPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['security_policy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_rule._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['securityPolicy'] = 'security_policy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_rule._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('priority',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'securityPolicy' in jsonified_request
    assert jsonified_request['securityPolicy'] == 'security_policy_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.remove_rule_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_rule_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_rule._get_unset_required_fields({})
    assert set(unset_fields) == set(('priority',)) & set(('project', 'securityPolicy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_rule_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_remove_rule') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_remove_rule') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.RemoveRuleSecurityPolicyRequest.pb(compute.RemoveRuleSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.RemoveRuleSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.remove_rule_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_rule_unary_rest_bad_request(transport: str='rest', request_type=compute.RemoveRuleSecurityPolicyRequest):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'security_policy': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_rule_unary(request)

def test_remove_rule_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'security_policy': 'sample2'}
        mock_args = dict(project='project_value', security_policy='security_policy_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.remove_rule_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{security_policy}/removeRule' % client.transport._host, args[1])

def test_remove_rule_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.remove_rule_unary(compute.RemoveRuleSecurityPolicyRequest(), project='project_value', security_policy='security_policy_value')

def test_remove_rule_unary_rest_error():
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetLabelsSecurityPolicyRequest, dict])
def test_set_labels_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request_init['global_set_labels_request_resource'] = {'label_fingerprint': 'label_fingerprint_value', 'labels': {}}
    test_field = compute.SetLabelsSecurityPolicyRequest.meta.fields['global_set_labels_request_resource']

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
    for (field, value) in request_init['global_set_labels_request_resource'].items():
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
                for i in range(0, len(request_init['global_set_labels_request_resource'][field])):
                    del request_init['global_set_labels_request_resource'][field][i][subfield]
            else:
                del request_init['global_set_labels_request_resource'][field][subfield]
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

def test_set_labels_rest_required_fields(request_type=compute.SetLabelsSecurityPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_labels._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('globalSetLabelsRequestResource', 'project', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_labels_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_set_labels') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_set_labels') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetLabelsSecurityPolicyRequest.pb(compute.SetLabelsSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetLabelsSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_labels(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_labels_rest_bad_request(transport: str='rest', request_type=compute.SetLabelsSecurityPolicyRequest):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_labels(request)

def test_set_labels_rest_flattened():
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'resource': 'sample2'}
        mock_args = dict(project='project_value', resource='resource_value', global_set_labels_request_resource=compute.GlobalSetLabelsRequest(label_fingerprint='label_fingerprint_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{resource}/setLabels' % client.transport._host, args[1])

def test_set_labels_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_labels(compute.SetLabelsSecurityPolicyRequest(), project='project_value', resource='resource_value', global_set_labels_request_resource=compute.GlobalSetLabelsRequest(label_fingerprint='label_fingerprint_value'))

def test_set_labels_rest_error():
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetLabelsSecurityPolicyRequest, dict])
def test_set_labels_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request_init['global_set_labels_request_resource'] = {'label_fingerprint': 'label_fingerprint_value', 'labels': {}}
    test_field = compute.SetLabelsSecurityPolicyRequest.meta.fields['global_set_labels_request_resource']

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
    for (field, value) in request_init['global_set_labels_request_resource'].items():
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
                for i in range(0, len(request_init['global_set_labels_request_resource'][field])):
                    del request_init['global_set_labels_request_resource'][field][i][subfield]
            else:
                del request_init['global_set_labels_request_resource'][field][subfield]
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

def test_set_labels_unary_rest_required_fields(request_type=compute.SetLabelsSecurityPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.SecurityPoliciesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_labels._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('globalSetLabelsRequestResource', 'project', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_labels_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SecurityPoliciesRestInterceptor())
    client = SecurityPoliciesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'post_set_labels') as post, mock.patch.object(transports.SecurityPoliciesRestInterceptor, 'pre_set_labels') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetLabelsSecurityPolicyRequest.pb(compute.SetLabelsSecurityPolicyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetLabelsSecurityPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_labels_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_labels_unary_rest_bad_request(transport: str='rest', request_type=compute.SetLabelsSecurityPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_labels_unary(request)

def test_set_labels_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'resource': 'sample2'}
        mock_args = dict(project='project_value', resource='resource_value', global_set_labels_request_resource=compute.GlobalSetLabelsRequest(label_fingerprint='label_fingerprint_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/global/securityPolicies/{resource}/setLabels' % client.transport._host, args[1])

def test_set_labels_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_labels_unary(compute.SetLabelsSecurityPolicyRequest(), project='project_value', resource='resource_value', global_set_labels_request_resource=compute.GlobalSetLabelsRequest(label_fingerprint='label_fingerprint_value'))

def test_set_labels_unary_rest_error():
    if False:
        return 10
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecurityPoliciesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SecurityPoliciesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SecurityPoliciesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SecurityPoliciesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.SecurityPoliciesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    client = SecurityPoliciesClient(transport=transport)
    assert client.transport is transport

@pytest.mark.parametrize('transport_class', [transports.SecurityPoliciesRestTransport])
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
    transport = SecurityPoliciesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_security_policies_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.SecurityPoliciesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_security_policies_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.compute_v1.services.security_policies.transports.SecurityPoliciesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.SecurityPoliciesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('add_rule', 'aggregated_list', 'delete', 'get', 'get_rule', 'insert', 'list', 'list_preconfigured_expression_sets', 'patch', 'patch_rule', 'remove_rule', 'set_labels')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_security_policies_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.compute_v1.services.security_policies.transports.SecurityPoliciesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SecurityPoliciesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_security_policies_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.compute_v1.services.security_policies.transports.SecurityPoliciesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SecurityPoliciesTransport()
        adc.assert_called_once()

def test_security_policies_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        SecurityPoliciesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

def test_security_policies_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.SecurityPoliciesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['rest'])
def test_security_policies_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_security_policies_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_security_policies_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = SecurityPoliciesClient(credentials=creds1, transport=transport_name)
    client2 = SecurityPoliciesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.add_rule._session
    session2 = client2.transport.add_rule._session
    assert session1 != session2
    session1 = client1.transport.aggregated_list._session
    session2 = client2.transport.aggregated_list._session
    assert session1 != session2
    session1 = client1.transport.delete._session
    session2 = client2.transport.delete._session
    assert session1 != session2
    session1 = client1.transport.get._session
    session2 = client2.transport.get._session
    assert session1 != session2
    session1 = client1.transport.get_rule._session
    session2 = client2.transport.get_rule._session
    assert session1 != session2
    session1 = client1.transport.insert._session
    session2 = client2.transport.insert._session
    assert session1 != session2
    session1 = client1.transport.list._session
    session2 = client2.transport.list._session
    assert session1 != session2
    session1 = client1.transport.list_preconfigured_expression_sets._session
    session2 = client2.transport.list_preconfigured_expression_sets._session
    assert session1 != session2
    session1 = client1.transport.patch._session
    session2 = client2.transport.patch._session
    assert session1 != session2
    session1 = client1.transport.patch_rule._session
    session2 = client2.transport.patch_rule._session
    assert session1 != session2
    session1 = client1.transport.remove_rule._session
    session2 = client2.transport.remove_rule._session
    assert session1 != session2
    session1 = client1.transport.set_labels._session
    session2 = client2.transport.set_labels._session
    assert session1 != session2

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = SecurityPoliciesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'clam'}
    path = SecurityPoliciesClient.common_billing_account_path(**expected)
    actual = SecurityPoliciesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = SecurityPoliciesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'octopus'}
    path = SecurityPoliciesClient.common_folder_path(**expected)
    actual = SecurityPoliciesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = SecurityPoliciesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = SecurityPoliciesClient.common_organization_path(**expected)
    actual = SecurityPoliciesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = SecurityPoliciesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel'}
    path = SecurityPoliciesClient.common_project_path(**expected)
    actual = SecurityPoliciesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = SecurityPoliciesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = SecurityPoliciesClient.common_location_path(**expected)
    actual = SecurityPoliciesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.SecurityPoliciesTransport, '_prep_wrapped_messages') as prep:
        client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.SecurityPoliciesTransport, '_prep_wrapped_messages') as prep:
        transport_class = SecurityPoliciesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session'}
    for (transport, close_name) in transports.items():
        client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = SecurityPoliciesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(SecurityPoliciesClient, transports.SecurityPoliciesRestTransport)])
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