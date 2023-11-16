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

def build_register_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    content_type = kwargs.pop('content_type', None)
    auto_version = kwargs.pop('auto_version', True)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _query_parameters = kwargs.pop('params', {})
    if auto_version is not None:
        _query_parameters['autoVersion'] = _SERIALIZER.query('auto_version', auto_version, 'bool')
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_query_parameters, headers=_header_parameters, **kwargs)

def build_list_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        print('Hello World!')
    name = kwargs.pop('name', None)
    tag = kwargs.pop('tag', None)
    version = kwargs.pop('version', None)
    framework = kwargs.pop('framework', None)
    description = kwargs.pop('description', None)
    count = kwargs.pop('count', None)
    offset = kwargs.pop('offset', None)
    skip_token = kwargs.pop('skip_token', None)
    tags = kwargs.pop('tags', None)
    properties = kwargs.pop('properties', None)
    run_id = kwargs.pop('run_id', None)
    dataset_id = kwargs.pop('dataset_id', None)
    order_by = kwargs.pop('order_by', None)
    latest_version_only = kwargs.pop('latest_version_only', False)
    feed = kwargs.pop('feed', None)
    list_view_type = kwargs.pop('list_view_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _query_parameters = kwargs.pop('params', {})
    if name is not None:
        _query_parameters['name'] = _SERIALIZER.query('name', name, 'str')
    if tag is not None:
        _query_parameters['tag'] = _SERIALIZER.query('tag', tag, 'str')
    if version is not None:
        _query_parameters['version'] = _SERIALIZER.query('version', version, 'str')
    if framework is not None:
        _query_parameters['framework'] = _SERIALIZER.query('framework', framework, 'str')
    if description is not None:
        _query_parameters['description'] = _SERIALIZER.query('description', description, 'str')
    if count is not None:
        _query_parameters['count'] = _SERIALIZER.query('count', count, 'int')
    if offset is not None:
        _query_parameters['offset'] = _SERIALIZER.query('offset', offset, 'int')
    if skip_token is not None:
        _query_parameters['$skipToken'] = _SERIALIZER.query('skip_token', skip_token, 'str')
    if tags is not None:
        _query_parameters['tags'] = _SERIALIZER.query('tags', tags, 'str')
    if properties is not None:
        _query_parameters['properties'] = _SERIALIZER.query('properties', properties, 'str')
    if run_id is not None:
        _query_parameters['runId'] = _SERIALIZER.query('run_id', run_id, 'str')
    if dataset_id is not None:
        _query_parameters['datasetId'] = _SERIALIZER.query('dataset_id', dataset_id, 'str')
    if order_by is not None:
        _query_parameters['orderBy'] = _SERIALIZER.query('order_by', order_by, 'str')
    if latest_version_only is not None:
        _query_parameters['latestVersionOnly'] = _SERIALIZER.query('latest_version_only', latest_version_only, 'bool')
    if feed is not None:
        _query_parameters['feed'] = _SERIALIZER.query('feed', feed, 'str')
    if list_view_type is not None:
        _query_parameters['listViewType'] = _SERIALIZER.query('list_view_type', list_view_type, 'str')
    _header_parameters = kwargs.pop('headers', {})
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_query_parameters, headers=_header_parameters, **kwargs)

def build_create_unregistered_input_model_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        return 10
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/createUnregisteredInput')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, headers=_header_parameters, **kwargs)

def build_create_unregistered_output_model_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/createUnregisteredOutput')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, headers=_header_parameters, **kwargs)

def build_batch_get_resolved_uris_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        print('Hello World!')
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/batchGetResolvedUris')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, headers=_header_parameters, **kwargs)

def build_query_by_id_request(id, subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        i = 10
        return i + 15
    include_deployment_settings = kwargs.pop('include_deployment_settings', False)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/{id}')
    path_format_arguments = {'id': _SERIALIZER.url('id', id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _query_parameters = kwargs.pop('params', {})
    if include_deployment_settings is not None:
        _query_parameters['includeDeploymentSettings'] = _SERIALIZER.query('include_deployment_settings', include_deployment_settings, 'bool')
    _header_parameters = kwargs.pop('headers', {})
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_query_parameters, headers=_header_parameters, **kwargs)

def build_delete_request(id, subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        while True:
            i = 10
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/{id}')
    path_format_arguments = {'id': _SERIALIZER.url('id', id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    return HttpRequest(method='DELETE', url=_url, **kwargs)

def build_patch_request(id, subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/{id}')
    path_format_arguments = {'id': _SERIALIZER.url('id', id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, headers=_header_parameters, **kwargs)

def build_list_query_post_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        while True:
            i = 10
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/list')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, headers=_header_parameters, **kwargs)

def build_batch_query_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        print('Hello World!')
    content_type = kwargs.pop('content_type', None)
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/querybatch')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, headers=_header_parameters, **kwargs)

def build_deployment_settings_request(subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        i = 10
        return i + 15
    content_type = kwargs.pop('content_type', None)
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/deploymentSettings')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    return HttpRequest(method='POST', url=_url, headers=_header_parameters, **kwargs)

class ModelsOperations(object):
    """ModelsOperations operations.

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
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def register(self, subscription_id, resource_group_name, workspace_name, body, auto_version=True, **kwargs):
        if False:
            i = 10
            return i + 15
        'register.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.Model\n        :param auto_version:\n        :type auto_version: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Model, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Model\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        _json = self._serialize.body(body, 'Model')
        request = build_register_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, auto_version=auto_version, template_url=self.register.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Model', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    register.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models'}

    @distributed_trace
    def list(self, subscription_id, resource_group_name, workspace_name, name=None, tag=None, version=None, framework=None, description=None, count=None, offset=None, skip_token=None, tags=None, properties=None, run_id=None, dataset_id=None, order_by=None, latest_version_only=False, feed=None, list_view_type=None, **kwargs):
        if False:
            while True:
                i = 10
        'list.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param name:\n        :type name: str\n        :param tag:\n        :type tag: str\n        :param version:\n        :type version: str\n        :param framework:\n        :type framework: str\n        :param description:\n        :type description: str\n        :param count:\n        :type count: int\n        :param offset:\n        :type offset: int\n        :param skip_token:\n        :type skip_token: str\n        :param tags:\n        :type tags: str\n        :param properties:\n        :type properties: str\n        :param run_id:\n        :type run_id: str\n        :param dataset_id:\n        :type dataset_id: str\n        :param order_by:\n        :type order_by: str\n        :param latest_version_only:\n        :type latest_version_only: bool\n        :param feed:\n        :type feed: str\n        :param list_view_type:\n        :type list_view_type: str or ~azure.mgmt.machinelearningservices.models.ListViewType\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ModelPagedResponse, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.ModelPagedResponse\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        request = build_list_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, name=name, tag=tag, version=version, framework=framework, description=description, count=count, offset=offset, skip_token=skip_token, tags=tags, properties=properties, run_id=run_id, dataset_id=dataset_id, order_by=order_by, latest_version_only=latest_version_only, feed=feed, list_view_type=list_view_type, template_url=self.list.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ModelPagedResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models'}

    @distributed_trace
    def create_unregistered_input_model(self, subscription_id, resource_group_name, workspace_name, body, **kwargs):
        if False:
            while True:
                i = 10
        'create_unregistered_input_model.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.CreateUnregisteredInputModelDto\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Model, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Model\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        _json = self._serialize.body(body, 'CreateUnregisteredInputModelDto')
        request = build_create_unregistered_input_model_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.create_unregistered_input_model.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Model', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create_unregistered_input_model.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/createUnregisteredInput'}

    @distributed_trace
    def create_unregistered_output_model(self, subscription_id, resource_group_name, workspace_name, body, **kwargs):
        if False:
            return 10
        'create_unregistered_output_model.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.CreateUnregisteredOutputModelDto\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Model, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Model\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        _json = self._serialize.body(body, 'CreateUnregisteredOutputModelDto')
        request = build_create_unregistered_output_model_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.create_unregistered_output_model.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Model', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create_unregistered_output_model.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/createUnregisteredOutput'}

    @distributed_trace
    def batch_get_resolved_uris(self, subscription_id, resource_group_name, workspace_name, body=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'batch_get_resolved_uris.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.BatchGetResolvedUrisDto\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: BatchModelPathResponseDto, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.BatchModelPathResponseDto\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        if body is not None:
            _json = self._serialize.body(body, 'BatchGetResolvedUrisDto')
        else:
            _json = None
        request = build_batch_get_resolved_uris_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.batch_get_resolved_uris.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('BatchModelPathResponseDto', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    batch_get_resolved_uris.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/batchGetResolvedUris'}

    @distributed_trace
    def query_by_id(self, id, subscription_id, resource_group_name, workspace_name, include_deployment_settings=False, **kwargs):
        if False:
            i = 10
            return i + 15
        'query_by_id.\n\n        :param id:\n        :type id: str\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param include_deployment_settings:\n        :type include_deployment_settings: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Model, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Model\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        request = build_query_by_id_request(id=id, subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, include_deployment_settings=include_deployment_settings, template_url=self.query_by_id.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Model', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    query_by_id.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/{id}'}

    @distributed_trace
    def delete(self, id, subscription_id, resource_group_name, workspace_name, **kwargs):
        if False:
            i = 10
            return i + 15
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
    delete.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/{id}'}

    @distributed_trace
    def patch(self, id, subscription_id, resource_group_name, workspace_name, body, **kwargs):
        if False:
            print('Hello World!')
        'patch.\n\n        :param id:\n        :type id: str\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: list[~azure.mgmt.machinelearningservices.models.Operation]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Model, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.Model\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
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
        deserialized = self._deserialize('Model', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    patch.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/{id}'}

    @distributed_trace
    def list_query_post(self, subscription_id, resource_group_name, workspace_name, body=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'list_query_post.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.ListModelsRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ModelListModelsRequestPagedResponse, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.ModelListModelsRequestPagedResponse\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        if body is not None:
            _json = self._serialize.body(body, 'ListModelsRequest')
        else:
            _json = None
        request = build_list_query_post_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.list_query_post.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ModelListModelsRequestPagedResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_query_post.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/list'}

    @distributed_trace
    def batch_query(self, subscription_id, resource_group_name, workspace_name, body=None, **kwargs):
        if False:
            print('Hello World!')
        'batch_query.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.ModelBatchDto\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ModelBatchResponseDto, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.ModelBatchResponseDto\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        if body is not None:
            _json = self._serialize.body(body, 'ModelBatchDto')
        else:
            _json = None
        request = build_batch_query_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.batch_query.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ModelBatchResponseDto', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    batch_query.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/querybatch'}

    @distributed_trace
    def deployment_settings(self, subscription_id, resource_group_name, workspace_name, body=None, **kwargs):
        if False:
            print('Hello World!')
        'deployment_settings.\n\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :param body:\n        :type body: ~azure.mgmt.machinelearningservices.models.ModelSettingsIdentifiers\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        content_type = kwargs.pop('content_type', 'application/json-patch+json')
        if body is not None:
            _json = self._serialize.body(body, 'ModelSettingsIdentifiers')
        else:
            _json = None
        request = build_deployment_settings_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, content_type=content_type, json=_json, template_url=self.deployment_settings.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    deployment_settings.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/models/deploymentSettings'}