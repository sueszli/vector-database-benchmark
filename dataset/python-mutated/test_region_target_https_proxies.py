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
from google.cloud.compute_v1.services.region_target_https_proxies import RegionTargetHttpsProxiesClient, pagers, transports
from google.cloud.compute_v1.types import compute

def client_cert_source_callback():
    if False:
        return 10
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
    assert RegionTargetHttpsProxiesClient._get_default_mtls_endpoint(None) is None
    assert RegionTargetHttpsProxiesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert RegionTargetHttpsProxiesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert RegionTargetHttpsProxiesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert RegionTargetHttpsProxiesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert RegionTargetHttpsProxiesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(RegionTargetHttpsProxiesClient, 'rest')])
def test_region_target_https_proxies_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.RegionTargetHttpsProxiesRestTransport, 'rest')])
def test_region_target_https_proxies_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(RegionTargetHttpsProxiesClient, 'rest')])
def test_region_target_https_proxies_client_from_service_account_file(client_class, transport_name):
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

def test_region_target_https_proxies_client_get_transport_class():
    if False:
        return 10
    transport = RegionTargetHttpsProxiesClient.get_transport_class()
    available_transports = [transports.RegionTargetHttpsProxiesRestTransport]
    assert transport in available_transports
    transport = RegionTargetHttpsProxiesClient.get_transport_class('rest')
    assert transport == transports.RegionTargetHttpsProxiesRestTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionTargetHttpsProxiesClient, transports.RegionTargetHttpsProxiesRestTransport, 'rest')])
@mock.patch.object(RegionTargetHttpsProxiesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionTargetHttpsProxiesClient))
def test_region_target_https_proxies_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(RegionTargetHttpsProxiesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(RegionTargetHttpsProxiesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(RegionTargetHttpsProxiesClient, transports.RegionTargetHttpsProxiesRestTransport, 'rest', 'true'), (RegionTargetHttpsProxiesClient, transports.RegionTargetHttpsProxiesRestTransport, 'rest', 'false')])
@mock.patch.object(RegionTargetHttpsProxiesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionTargetHttpsProxiesClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_region_target_https_proxies_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [RegionTargetHttpsProxiesClient])
@mock.patch.object(RegionTargetHttpsProxiesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionTargetHttpsProxiesClient))
def test_region_target_https_proxies_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionTargetHttpsProxiesClient, transports.RegionTargetHttpsProxiesRestTransport, 'rest')])
def test_region_target_https_proxies_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RegionTargetHttpsProxiesClient, transports.RegionTargetHttpsProxiesRestTransport, 'rest', None)])
def test_region_target_https_proxies_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('request_type', [compute.DeleteRegionTargetHttpsProxyRequest, dict])
def test_delete_rest(request_type):
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
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

def test_delete_rest_required_fields(request_type=compute.DeleteRegionTargetHttpsProxyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionTargetHttpsProxyRequest.pb(compute.DeleteRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionTargetHttpsProxyRequest):
    if False:
        i = 10
        return i + 15
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete(request)

def test_delete_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}' % client.transport._host, args[1])

def test_delete_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete(compute.DeleteRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')

def test_delete_rest_error():
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteRegionTargetHttpsProxyRequest, dict])
def test_delete_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
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

def test_delete_unary_rest_required_fields(request_type=compute.DeleteRegionTargetHttpsProxyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionTargetHttpsProxyRequest.pb(compute.DeleteRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionTargetHttpsProxyRequest):
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_unary(request)

def test_delete_unary_rest_flattened():
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}' % client.transport._host, args[1])

def test_delete_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_unary(compute.DeleteRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')

def test_delete_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetRegionTargetHttpsProxyRequest, dict])
def test_get_rest(request_type):
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.TargetHttpsProxy(authorization_policy='authorization_policy_value', certificate_map='certificate_map_value', creation_timestamp='creation_timestamp_value', description='description_value', fingerprint='fingerprint_value', http_keep_alive_timeout_sec=2868, id=205, kind='kind_value', name='name_value', proxy_bind=True, quic_override='quic_override_value', region='region_value', self_link='self_link_value', server_tls_policy='server_tls_policy_value', ssl_certificates=['ssl_certificates_value'], ssl_policy='ssl_policy_value', url_map='url_map_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.TargetHttpsProxy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get(request)
    assert isinstance(response, compute.TargetHttpsProxy)
    assert response.authorization_policy == 'authorization_policy_value'
    assert response.certificate_map == 'certificate_map_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.fingerprint == 'fingerprint_value'
    assert response.http_keep_alive_timeout_sec == 2868
    assert response.id == 205
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.proxy_bind is True
    assert response.quic_override == 'quic_override_value'
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.server_tls_policy == 'server_tls_policy_value'
    assert response.ssl_certificates == ['ssl_certificates_value']
    assert response.ssl_policy == 'ssl_policy_value'
    assert response.url_map == 'url_map_value'

def test_get_rest_required_fields(request_type=compute.GetRegionTargetHttpsProxyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.TargetHttpsProxy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.TargetHttpsProxy.pb(return_value)
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
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('project', 'region', 'targetHttpsProxy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_get') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_get') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetRegionTargetHttpsProxyRequest.pb(compute.GetRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.TargetHttpsProxy.to_json(compute.TargetHttpsProxy())
        request = compute.GetRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.TargetHttpsProxy()
        client.get(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rest_bad_request(transport: str='rest', request_type=compute.GetRegionTargetHttpsProxyRequest):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
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
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.TargetHttpsProxy()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.TargetHttpsProxy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}' % client.transport._host, args[1])

def test_get_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get(compute.GetRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')

def test_get_rest_error():
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionTargetHttpsProxyRequest, dict])
def test_insert_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['target_https_proxy_resource'] = {'authorization_policy': 'authorization_policy_value', 'certificate_map': 'certificate_map_value', 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'http_keep_alive_timeout_sec': 2868, 'id': 205, 'kind': 'kind_value', 'name': 'name_value', 'proxy_bind': True, 'quic_override': 'quic_override_value', 'region': 'region_value', 'self_link': 'self_link_value', 'server_tls_policy': 'server_tls_policy_value', 'ssl_certificates': ['ssl_certificates_value1', 'ssl_certificates_value2'], 'ssl_policy': 'ssl_policy_value', 'url_map': 'url_map_value'}
    test_field = compute.InsertRegionTargetHttpsProxyRequest.meta.fields['target_https_proxy_resource']

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
    for (field, value) in request_init['target_https_proxy_resource'].items():
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
                for i in range(0, len(request_init['target_https_proxy_resource'][field])):
                    del request_init['target_https_proxy_resource'][field][i][subfield]
            else:
                del request_init['target_https_proxy_resource'][field][subfield]
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

def test_insert_rest_required_fields(request_type=compute.InsertRegionTargetHttpsProxyRequest):
    if False:
        return 10
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
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
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionTargetHttpsProxyRequest.pb(compute.InsertRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionTargetHttpsProxyRequest):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies' % client.transport._host, args[1])

def test_insert_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert(compute.InsertRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))

def test_insert_rest_error():
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionTargetHttpsProxyRequest, dict])
def test_insert_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['target_https_proxy_resource'] = {'authorization_policy': 'authorization_policy_value', 'certificate_map': 'certificate_map_value', 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'http_keep_alive_timeout_sec': 2868, 'id': 205, 'kind': 'kind_value', 'name': 'name_value', 'proxy_bind': True, 'quic_override': 'quic_override_value', 'region': 'region_value', 'self_link': 'self_link_value', 'server_tls_policy': 'server_tls_policy_value', 'ssl_certificates': ['ssl_certificates_value1', 'ssl_certificates_value2'], 'ssl_policy': 'ssl_policy_value', 'url_map': 'url_map_value'}
    test_field = compute.InsertRegionTargetHttpsProxyRequest.meta.fields['target_https_proxy_resource']

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
    for (field, value) in request_init['target_https_proxy_resource'].items():
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
                for i in range(0, len(request_init['target_https_proxy_resource'][field])):
                    del request_init['target_https_proxy_resource'][field][i][subfield]
            else:
                del request_init['target_https_proxy_resource'][field][subfield]
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

def test_insert_unary_rest_required_fields(request_type=compute.InsertRegionTargetHttpsProxyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
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
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionTargetHttpsProxyRequest.pb(compute.InsertRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionTargetHttpsProxyRequest):
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies' % client.transport._host, args[1])

def test_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert_unary(compute.InsertRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))

def test_insert_unary_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ListRegionTargetHttpsProxiesRequest, dict])
def test_list_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.TargetHttpsProxyList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.TargetHttpsProxyList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list(request)
    assert isinstance(response, pagers.ListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'

def test_list_rest_required_fields(request_type=compute.ListRegionTargetHttpsProxiesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
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
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.TargetHttpsProxyList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.TargetHttpsProxyList.pb(return_value)
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
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_list') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListRegionTargetHttpsProxiesRequest.pb(compute.ListRegionTargetHttpsProxiesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.TargetHttpsProxyList.to_json(compute.TargetHttpsProxyList())
        request = compute.ListRegionTargetHttpsProxiesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.TargetHttpsProxyList()
        client.list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rest_bad_request(transport: str='rest', request_type=compute.ListRegionTargetHttpsProxiesRequest):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.TargetHttpsProxyList()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.TargetHttpsProxyList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies' % client.transport._host, args[1])

def test_list_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list(compute.ListRegionTargetHttpsProxiesRequest(), project='project_value', region='region_value')

def test_list_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.TargetHttpsProxyList(items=[compute.TargetHttpsProxy(), compute.TargetHttpsProxy(), compute.TargetHttpsProxy()], next_page_token='abc'), compute.TargetHttpsProxyList(items=[], next_page_token='def'), compute.TargetHttpsProxyList(items=[compute.TargetHttpsProxy()], next_page_token='ghi'), compute.TargetHttpsProxyList(items=[compute.TargetHttpsProxy(), compute.TargetHttpsProxy()]))
        response = response + response
        response = tuple((compute.TargetHttpsProxyList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        pager = client.list(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.TargetHttpsProxy) for i in results))
        pages = list(client.list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.PatchRegionTargetHttpsProxyRequest, dict])
def test_patch_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request_init['target_https_proxy_resource'] = {'authorization_policy': 'authorization_policy_value', 'certificate_map': 'certificate_map_value', 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'http_keep_alive_timeout_sec': 2868, 'id': 205, 'kind': 'kind_value', 'name': 'name_value', 'proxy_bind': True, 'quic_override': 'quic_override_value', 'region': 'region_value', 'self_link': 'self_link_value', 'server_tls_policy': 'server_tls_policy_value', 'ssl_certificates': ['ssl_certificates_value1', 'ssl_certificates_value2'], 'ssl_policy': 'ssl_policy_value', 'url_map': 'url_map_value'}
    test_field = compute.PatchRegionTargetHttpsProxyRequest.meta.fields['target_https_proxy_resource']

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
    for (field, value) in request_init['target_https_proxy_resource'].items():
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
                for i in range(0, len(request_init['target_https_proxy_resource'][field])):
                    del request_init['target_https_proxy_resource'][field][i][subfield]
            else:
                del request_init['target_https_proxy_resource'][field][subfield]
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

def test_patch_rest_required_fields(request_type=compute.PatchRegionTargetHttpsProxyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxy', 'targetHttpsProxyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_patch') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchRegionTargetHttpsProxyRequest.pb(compute.PatchRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_rest_bad_request(transport: str='rest', request_type=compute.PatchRegionTargetHttpsProxyRequest):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch(request)

def test_patch_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}' % client.transport._host, args[1])

def test_patch_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch(compute.PatchRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))

def test_patch_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchRegionTargetHttpsProxyRequest, dict])
def test_patch_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request_init['target_https_proxy_resource'] = {'authorization_policy': 'authorization_policy_value', 'certificate_map': 'certificate_map_value', 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'http_keep_alive_timeout_sec': 2868, 'id': 205, 'kind': 'kind_value', 'name': 'name_value', 'proxy_bind': True, 'quic_override': 'quic_override_value', 'region': 'region_value', 'self_link': 'self_link_value', 'server_tls_policy': 'server_tls_policy_value', 'ssl_certificates': ['ssl_certificates_value1', 'ssl_certificates_value2'], 'ssl_policy': 'ssl_policy_value', 'url_map': 'url_map_value'}
    test_field = compute.PatchRegionTargetHttpsProxyRequest.meta.fields['target_https_proxy_resource']

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
    for (field, value) in request_init['target_https_proxy_resource'].items():
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
                for i in range(0, len(request_init['target_https_proxy_resource'][field])):
                    del request_init['target_https_proxy_resource'][field][i][subfield]
            else:
                del request_init['target_https_proxy_resource'][field][subfield]
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

def test_patch_unary_rest_required_fields(request_type=compute.PatchRegionTargetHttpsProxyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxy', 'targetHttpsProxyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_patch') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchRegionTargetHttpsProxyRequest.pb(compute.PatchRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_unary_rest_bad_request(transport: str='rest', request_type=compute.PatchRegionTargetHttpsProxyRequest):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
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
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}' % client.transport._host, args[1])

def test_patch_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_unary(compute.PatchRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', target_https_proxy_resource=compute.TargetHttpsProxy(authorization_policy='authorization_policy_value'))

def test_patch_unary_rest_error():
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetSslCertificatesRegionTargetHttpsProxyRequest, dict])
def test_set_ssl_certificates_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request_init['region_target_https_proxies_set_ssl_certificates_request_resource'] = {'ssl_certificates': ['ssl_certificates_value1', 'ssl_certificates_value2']}
    test_field = compute.SetSslCertificatesRegionTargetHttpsProxyRequest.meta.fields['region_target_https_proxies_set_ssl_certificates_request_resource']

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
    for (field, value) in request_init['region_target_https_proxies_set_ssl_certificates_request_resource'].items():
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
                for i in range(0, len(request_init['region_target_https_proxies_set_ssl_certificates_request_resource'][field])):
                    del request_init['region_target_https_proxies_set_ssl_certificates_request_resource'][field][i][subfield]
            else:
                del request_init['region_target_https_proxies_set_ssl_certificates_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_ssl_certificates(request)
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

def test_set_ssl_certificates_rest_required_fields(request_type=compute.SetSslCertificatesRegionTargetHttpsProxyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_ssl_certificates._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_ssl_certificates._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_ssl_certificates(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_ssl_certificates_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_ssl_certificates._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'regionTargetHttpsProxiesSetSslCertificatesRequestResource', 'targetHttpsProxy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_ssl_certificates_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_set_ssl_certificates') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_set_ssl_certificates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetSslCertificatesRegionTargetHttpsProxyRequest.pb(compute.SetSslCertificatesRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetSslCertificatesRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_ssl_certificates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_ssl_certificates_rest_bad_request(transport: str='rest', request_type=compute.SetSslCertificatesRegionTargetHttpsProxyRequest):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_ssl_certificates(request)

def test_set_ssl_certificates_rest_flattened():
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', region_target_https_proxies_set_ssl_certificates_request_resource=compute.RegionTargetHttpsProxiesSetSslCertificatesRequest(ssl_certificates=['ssl_certificates_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_ssl_certificates(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}/setSslCertificates' % client.transport._host, args[1])

def test_set_ssl_certificates_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_ssl_certificates(compute.SetSslCertificatesRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', region_target_https_proxies_set_ssl_certificates_request_resource=compute.RegionTargetHttpsProxiesSetSslCertificatesRequest(ssl_certificates=['ssl_certificates_value']))

def test_set_ssl_certificates_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetSslCertificatesRegionTargetHttpsProxyRequest, dict])
def test_set_ssl_certificates_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request_init['region_target_https_proxies_set_ssl_certificates_request_resource'] = {'ssl_certificates': ['ssl_certificates_value1', 'ssl_certificates_value2']}
    test_field = compute.SetSslCertificatesRegionTargetHttpsProxyRequest.meta.fields['region_target_https_proxies_set_ssl_certificates_request_resource']

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
    for (field, value) in request_init['region_target_https_proxies_set_ssl_certificates_request_resource'].items():
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
                for i in range(0, len(request_init['region_target_https_proxies_set_ssl_certificates_request_resource'][field])):
                    del request_init['region_target_https_proxies_set_ssl_certificates_request_resource'][field][i][subfield]
            else:
                del request_init['region_target_https_proxies_set_ssl_certificates_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_ssl_certificates_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_ssl_certificates_unary_rest_required_fields(request_type=compute.SetSslCertificatesRegionTargetHttpsProxyRequest):
    if False:
        return 10
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_ssl_certificates._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_ssl_certificates._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_ssl_certificates_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_ssl_certificates_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_ssl_certificates._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'regionTargetHttpsProxiesSetSslCertificatesRequestResource', 'targetHttpsProxy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_ssl_certificates_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_set_ssl_certificates') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_set_ssl_certificates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetSslCertificatesRegionTargetHttpsProxyRequest.pb(compute.SetSslCertificatesRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetSslCertificatesRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_ssl_certificates_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_ssl_certificates_unary_rest_bad_request(transport: str='rest', request_type=compute.SetSslCertificatesRegionTargetHttpsProxyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_ssl_certificates_unary(request)

def test_set_ssl_certificates_unary_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', region_target_https_proxies_set_ssl_certificates_request_resource=compute.RegionTargetHttpsProxiesSetSslCertificatesRequest(ssl_certificates=['ssl_certificates_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_ssl_certificates_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}/setSslCertificates' % client.transport._host, args[1])

def test_set_ssl_certificates_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_ssl_certificates_unary(compute.SetSslCertificatesRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', region_target_https_proxies_set_ssl_certificates_request_resource=compute.RegionTargetHttpsProxiesSetSslCertificatesRequest(ssl_certificates=['ssl_certificates_value']))

def test_set_ssl_certificates_unary_rest_error():
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetUrlMapRegionTargetHttpsProxyRequest, dict])
def test_set_url_map_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request_init['url_map_reference_resource'] = {'url_map': 'url_map_value'}
    test_field = compute.SetUrlMapRegionTargetHttpsProxyRequest.meta.fields['url_map_reference_resource']

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
    for (field, value) in request_init['url_map_reference_resource'].items():
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
                for i in range(0, len(request_init['url_map_reference_resource'][field])):
                    del request_init['url_map_reference_resource'][field][i][subfield]
            else:
                del request_init['url_map_reference_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_url_map(request)
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

def test_set_url_map_rest_required_fields(request_type=compute.SetUrlMapRegionTargetHttpsProxyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_url_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_url_map._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_url_map(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_url_map_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_url_map._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxy', 'urlMapReferenceResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_url_map_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_set_url_map') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_set_url_map') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetUrlMapRegionTargetHttpsProxyRequest.pb(compute.SetUrlMapRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetUrlMapRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_url_map(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_url_map_rest_bad_request(transport: str='rest', request_type=compute.SetUrlMapRegionTargetHttpsProxyRequest):
    if False:
        i = 10
        return i + 15
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_url_map(request)

def test_set_url_map_rest_flattened():
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', url_map_reference_resource=compute.UrlMapReference(url_map='url_map_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_url_map(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}/setUrlMap' % client.transport._host, args[1])

def test_set_url_map_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_url_map(compute.SetUrlMapRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', url_map_reference_resource=compute.UrlMapReference(url_map='url_map_value'))

def test_set_url_map_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetUrlMapRegionTargetHttpsProxyRequest, dict])
def test_set_url_map_unary_rest(request_type):
    if False:
        return 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request_init['url_map_reference_resource'] = {'url_map': 'url_map_value'}
    test_field = compute.SetUrlMapRegionTargetHttpsProxyRequest.meta.fields['url_map_reference_resource']

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
    for (field, value) in request_init['url_map_reference_resource'].items():
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
                for i in range(0, len(request_init['url_map_reference_resource'][field])):
                    del request_init['url_map_reference_resource'][field][i][subfield]
            else:
                del request_init['url_map_reference_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_url_map_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_url_map_unary_rest_required_fields(request_type=compute.SetUrlMapRegionTargetHttpsProxyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionTargetHttpsProxiesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['target_https_proxy'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_url_map._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['targetHttpsProxy'] = 'target_https_proxy_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_url_map._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'targetHttpsProxy' in jsonified_request
    assert jsonified_request['targetHttpsProxy'] == 'target_https_proxy_value'
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_url_map_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_url_map_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_url_map._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'targetHttpsProxy', 'urlMapReferenceResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_url_map_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionTargetHttpsProxiesRestInterceptor())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'post_set_url_map') as post, mock.patch.object(transports.RegionTargetHttpsProxiesRestInterceptor, 'pre_set_url_map') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetUrlMapRegionTargetHttpsProxyRequest.pb(compute.SetUrlMapRegionTargetHttpsProxyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetUrlMapRegionTargetHttpsProxyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_url_map_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_url_map_unary_rest_bad_request(transport: str='rest', request_type=compute.SetUrlMapRegionTargetHttpsProxyRequest):
    if False:
        i = 10
        return i + 15
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_url_map_unary(request)

def test_set_url_map_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'target_https_proxy': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', url_map_reference_resource=compute.UrlMapReference(url_map='url_map_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_url_map_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/targetHttpsProxies/{target_https_proxy}/setUrlMap' % client.transport._host, args[1])

def test_set_url_map_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_url_map_unary(compute.SetUrlMapRegionTargetHttpsProxyRequest(), project='project_value', region='region_value', target_https_proxy='target_https_proxy_value', url_map_reference_resource=compute.UrlMapReference(url_map='url_map_value'))

def test_set_url_map_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionTargetHttpsProxiesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionTargetHttpsProxiesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionTargetHttpsProxiesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionTargetHttpsProxiesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionTargetHttpsProxiesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    client = RegionTargetHttpsProxiesClient(transport=transport)
    assert client.transport is transport

@pytest.mark.parametrize('transport_class', [transports.RegionTargetHttpsProxiesRestTransport])
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
        for i in range(10):
            print('nop')
    transport = RegionTargetHttpsProxiesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_region_target_https_proxies_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.RegionTargetHttpsProxiesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_region_target_https_proxies_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.compute_v1.services.region_target_https_proxies.transports.RegionTargetHttpsProxiesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.RegionTargetHttpsProxiesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('delete', 'get', 'insert', 'list', 'patch', 'set_ssl_certificates', 'set_url_map')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_region_target_https_proxies_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.compute_v1.services.region_target_https_proxies.transports.RegionTargetHttpsProxiesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionTargetHttpsProxiesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_region_target_https_proxies_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.compute_v1.services.region_target_https_proxies.transports.RegionTargetHttpsProxiesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionTargetHttpsProxiesTransport()
        adc.assert_called_once()

def test_region_target_https_proxies_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        RegionTargetHttpsProxiesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

def test_region_target_https_proxies_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.RegionTargetHttpsProxiesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_target_https_proxies_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_target_https_proxies_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_target_https_proxies_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = RegionTargetHttpsProxiesClient(credentials=creds1, transport=transport_name)
    client2 = RegionTargetHttpsProxiesClient(credentials=creds2, transport=transport_name)
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
    session1 = client1.transport.patch._session
    session2 = client2.transport.patch._session
    assert session1 != session2
    session1 = client1.transport.set_ssl_certificates._session
    session2 = client2.transport.set_ssl_certificates._session
    assert session1 != session2
    session1 = client1.transport.set_url_map._session
    session2 = client2.transport.set_url_map._session
    assert session1 != session2

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = RegionTargetHttpsProxiesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'clam'}
    path = RegionTargetHttpsProxiesClient.common_billing_account_path(**expected)
    actual = RegionTargetHttpsProxiesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = RegionTargetHttpsProxiesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = RegionTargetHttpsProxiesClient.common_folder_path(**expected)
    actual = RegionTargetHttpsProxiesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = RegionTargetHttpsProxiesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = RegionTargetHttpsProxiesClient.common_organization_path(**expected)
    actual = RegionTargetHttpsProxiesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = RegionTargetHttpsProxiesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'mussel'}
    path = RegionTargetHttpsProxiesClient.common_project_path(**expected)
    actual = RegionTargetHttpsProxiesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = RegionTargetHttpsProxiesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = RegionTargetHttpsProxiesClient.common_location_path(**expected)
    actual = RegionTargetHttpsProxiesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.RegionTargetHttpsProxiesTransport, '_prep_wrapped_messages') as prep:
        client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.RegionTargetHttpsProxiesTransport, '_prep_wrapped_messages') as prep:
        transport_class = RegionTargetHttpsProxiesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session'}
    for (transport, close_name) in transports.items():
        client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = RegionTargetHttpsProxiesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(RegionTargetHttpsProxiesClient, transports.RegionTargetHttpsProxiesRestTransport)])
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