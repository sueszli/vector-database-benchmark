from typing import TYPE_CHECKING
from msrest import Serializer
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from .._vendor import _convert_request, _format_url_section
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_create_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        return 10
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, headers=_header_parameters, **kwargs)

def build_list_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    run_id = kwargs.pop('run_id', None)
    project_id = kwargs.pop('project_id', None)
    name = kwargs.pop('name', None)
    tag = kwargs.pop('tag', None)
    count = kwargs.pop('count', None)
    skip_token = kwargs.pop('skip_token', None)
    tags = kwargs.pop('tags', None)
    properties = kwargs.pop('properties', None)
    type = kwargs.pop('type', None)
    orderby = kwargs.pop('orderby', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _query_parameters = kwargs.pop('params', {})
    if run_id is not None:
        _query_parameters['runId'] = _SERIALIZER.query('run_id', run_id, 'str')
    if project_id is not None:
        _query_parameters['projectId'] = _SERIALIZER.query('project_id', project_id, 'str')
    if name is not None:
        _query_parameters['name'] = _SERIALIZER.query('name', name, 'str')
    if tag is not None:
        _query_parameters['tag'] = _SERIALIZER.query('tag', tag, 'str')
    if count is not None:
        _query_parameters['count'] = _SERIALIZER.query('count', count, 'int')
    if skip_token is not None:
        _query_parameters['$skipToken'] = _SERIALIZER.query('skip_token', skip_token, 'str')
    if tags is not None:
        _query_parameters['tags'] = _SERIALIZER.query('tags', tags, 'str')
    if properties is not None:
        _query_parameters['properties'] = _SERIALIZER.query('properties', properties, 'str')
    if type is not None:
        _query_parameters['type'] = _SERIALIZER.query('type', type, 'str')
    if orderby is not None:
        _query_parameters['orderby'] = _SERIALIZER.query('orderby', orderby, 'str')
    _header_parameters = kwargs.pop('headers', {})
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_query_parameters, headers=_header_parameters, **kwargs)

def build_patch_request(id, subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        return 10
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets/{id}')
    path_format_arguments = {'id': _SERIALIZER.url('id', id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, headers=_header_parameters, **kwargs)

def build_delete_request(id, subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        while True:
            i = 10
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets/{id}')
    path_format_arguments = {'id': _SERIALIZER.url('id', id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    return HttpRequest(method='DELETE', url=_url, **kwargs)

def build_query_by_id_request(id, subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        i = 10
        return i + 15
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets/{id}')
    path_format_arguments = {'id': _SERIALIZER.url('id', id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, headers=_header_parameters, **kwargs)

class AssetsOperations(object):
    """AssetsOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~azure.mgmt.machinelearningservices.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            i = 10
            return i + 15
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def create(self, subscription_id, resource_group_name, workspace_name, body=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'create.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.Asset\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Asset, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Asset\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        if body is not None:
            _json = self._serialize.body(body, 'Asset')
        else:
            _json = None
        request = build_create_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.create.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Asset', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets'}

    @distributed_trace
    def list(self, subscription_id, resource_group_name, workspace_name, run_id=None, project_id=None, name=None, tag=None, count=None, skip_token=None, tags=None, properties=None, type=None, orderby=None, **kwargs):
        if False:
            print('Hello World!')
        'list.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param run_id:\n        :type run_id: str\n        :param project_id:\n        :type project_id: str\n        :param name:\n        :type name: str\n        :param tag:\n        :type tag: str\n        :param count:\n        :type count: int\n        :param skip_token:\n        :type skip_token: str\n        :param tags:\n        :type tags: str\n        :param properties:\n        :type properties: str\n        :param type:\n        :type type: str\n        :param orderby:\n        :type orderby: str or ~azure.mgmt.machinelearningservices.models.OrderString\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AssetPaginatedResult, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.AssetPaginatedResult\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        request = build_list_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, run_id=run_id, project_id=project_id, name=name, tag=tag, count=count, skip_token=skip_token, tags=tags, properties=properties, type=type, orderby=orderby, template_url=self.list.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('AssetPaginatedResult', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets'}

    @distributed_trace
    def patch(self, id, subscription_id, resource_group_name, workspace_name, body, **kwargs):
        if False:
            while True:
                i = 10
        'patch.\n\n        :param id:\n        :type id: str\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: list[~azure.mgmt.machinelearningservices.models.Operation]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Asset, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Asset\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        _json = self._serialize.body(body, '[Operation]')
        request = build_patch_request(id=id, subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.patch.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Asset', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    patch.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets/{id}'}

    @distributed_trace
    def delete(self, id, subscription_id, resource_group_name, workspace_name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'delete.\n\n        :param id:\n        :type id: str\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        request = build_delete_request(id=id, subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, template_url=self.delete.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets/{id}'}

    @distributed_trace
    def query_by_id(self, id, subscription_id, resource_group_name, workspace_name, **kwargs):
        if False:
            while True:
                i = 10
        'query_by_id.\n\n        :param id:\n        :type id: str\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Asset, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Asset\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        request = build_query_by_id_request(id=id, subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, template_url=self.query_by_id.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Asset', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    query_by_id.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/assets/{id}'}