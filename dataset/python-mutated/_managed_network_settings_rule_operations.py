from typing import Any, AsyncIterable, Callable, Dict, Optional, TypeVar, Union
from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import AsyncHttpResponse
from azure.core.polling import AsyncLROPoller, AsyncNoPolling, AsyncPollingMethod
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.tracing.decorator_async import distributed_trace_async
from azure.mgmt.core.exceptions import ARMErrorFormat
from azure.mgmt.core.polling.async_arm_polling import AsyncARMPolling
from ... import models as _models
from ..._vendor import _convert_request
from ...operations._managed_network_settings_rule_operations import build_create_or_update_request_initial, build_delete_request_initial, build_get_request, build_list_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class ManagedNetworkSettingsRuleOperations:
    """ManagedNetworkSettingsRuleOperations async operations.

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

    def __init__(self, client, config, serializer, deserializer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def list(self, resource_group_name: str, workspace_name: str, **kwargs: Any) -> AsyncIterable['_models.OutboundRuleListResult']:
        if False:
            for i in range(10):
                print('nop')
        'list.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param workspace_name: Azure Machine Learning Workspace Name.\n        :type workspace_name: str\n        :keyword api_version: Api Version. The default value is "2022-10-01". Note that overriding this\n         default value may result in unsupported behavior.\n        :paramtype api_version: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either OutboundRuleListResult or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.machinelearningservices.models.OutboundRuleListResult]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        api_version = kwargs.pop('api_version', '2022-10-01')
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_request(subscription_id=self._config.subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, api_version=api_version, template_url=self.list.metadata['url'])
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                request = build_list_request(subscription_id=self._config.subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, api_version=api_version, template_url=next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize('OutboundRuleListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, AsyncList(list_of_elem))

        async def get_next(next_link=None):
            request = prepare_request(next_link)
            pipeline_response = await self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return AsyncItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/outboundRules'}

    async def _delete_initial(self, resource_group_name: str, workspace_name: str, rule_name: str, **kwargs: Any) -> None:
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = kwargs.pop('api_version', '2022-10-01')
        request = build_delete_request_initial(subscription_id=self._config.subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, rule_name=rule_name, api_version=api_version, template_url=self._delete_initial.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = await self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _delete_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/outboundRules/{ruleName}'}

    @distributed_trace_async
    async def begin_delete(self, resource_group_name: str, workspace_name: str, rule_name: str, **kwargs: Any) -> AsyncLROPoller[None]:
        """delete.

        :param resource_group_name: The name of the resource group. The name is case insensitive.
        :type resource_group_name: str
        :param workspace_name: Azure Machine Learning Workspace Name.
        :type workspace_name: str
        :param rule_name:
        :type rule_name: str
        :keyword api_version: Api Version. The default value is "2022-10-01". Note that overriding this
         default value may result in unsupported behavior.
        :paramtype api_version: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :keyword str continuation_token: A continuation token to restart a poller from a saved state.
        :keyword polling: By default, your polling method will be AsyncARMPolling. Pass in False for
         this operation to not poll, or pass in your own initialized polling object for a personal
         polling strategy.
        :paramtype polling: bool or ~azure.core.polling.AsyncPollingMethod
        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
         Retry-After header is present.
        :return: An instance of AsyncLROPoller that returns either None or the result of cls(response)
        :rtype: ~azure.core.polling.AsyncLROPoller[None]
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        api_version = kwargs.pop('api_version', '2022-10-01')
        polling = kwargs.pop('polling', True)
        cls = kwargs.pop('cls', None)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = await self._delete_initial(resource_group_name=resource_group_name, workspace_name=workspace_name, rule_name=rule_name, api_version=api_version, cls=lambda x, y, z: x, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method = AsyncARMPolling(lro_delay, **kwargs)
        elif polling is False:
            polling_method = AsyncNoPolling()
        else:
            polling_method = polling
        if cont_token:
            return AsyncLROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return AsyncLROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/outboundRules/{ruleName}'}

    @distributed_trace_async
    async def get(self, resource_group_name: str, workspace_name: str, rule_name: str, **kwargs: Any) -> '_models.OutboundRuleBasicResource':
        """get.

        :param resource_group_name: The name of the resource group. The name is case insensitive.
        :type resource_group_name: str
        :param workspace_name: Azure Machine Learning Workspace Name.
        :type workspace_name: str
        :param rule_name:
        :type rule_name: str
        :keyword api_version: Api Version. The default value is "2022-10-01". Note that overriding this
         default value may result in unsupported behavior.
        :paramtype api_version: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: OutboundRuleBasicResource, or the result of cls(response)
        :rtype: ~azure.mgmt.machinelearningservices.models.OutboundRuleBasicResource
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = kwargs.pop('api_version', '2022-10-01')
        request = build_get_request(subscription_id=self._config.subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, rule_name=rule_name, api_version=api_version, template_url=self.get.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = await self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('OutboundRuleBasicResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/outboundRules/{ruleName}'}

    async def _create_or_update_initial(self, resource_group_name: str, workspace_name: str, rule_name: str, body: '_models.OutboundRuleBasicResource', **kwargs: Any) -> Optional['_models.OutboundRuleBasicResource']:
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = kwargs.pop('api_version', '2022-10-01')
        content_type = kwargs.pop('content_type', 'application/json')
        _json = self._serialize.body(body, 'OutboundRuleBasicResource')
        request = build_create_or_update_request_initial(subscription_id=self._config.subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, rule_name=rule_name, api_version=api_version, content_type=content_type, json=_json, template_url=self._create_or_update_initial.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = await self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('OutboundRuleBasicResource', pipeline_response)
        if response.status_code == 202:
            response_headers['Location'] = self._deserialize('str', response.headers.get('Location'))
            response_headers['Retry-After'] = self._deserialize('int', response.headers.get('Retry-After'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_or_update_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/outboundRules/{ruleName}'}

    @distributed_trace_async
    async def begin_create_or_update(self, resource_group_name: str, workspace_name: str, rule_name: str, body: '_models.OutboundRuleBasicResource', **kwargs: Any) -> AsyncLROPoller['_models.OutboundRuleBasicResource']:
        """create_or_update.

        :param resource_group_name: The name of the resource group. The name is case insensitive.
        :type resource_group_name: str
        :param workspace_name: Azure Machine Learning Workspace Name.
        :type workspace_name: str
        :param rule_name:
        :type rule_name: str
        :param body:
        :type body: ~azure.mgmt.machinelearningservices.models.OutboundRuleBasicResource
        :keyword api_version: Api Version. The default value is "2022-10-01". Note that overriding this
         default value may result in unsupported behavior.
        :paramtype api_version: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :keyword str continuation_token: A continuation token to restart a poller from a saved state.
        :keyword polling: By default, your polling method will be AsyncARMPolling. Pass in False for
         this operation to not poll, or pass in your own initialized polling object for a personal
         polling strategy.
        :paramtype polling: bool or ~azure.core.polling.AsyncPollingMethod
        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
         Retry-After header is present.
        :return: An instance of AsyncLROPoller that returns either OutboundRuleBasicResource or the
         result of cls(response)
        :rtype:
         ~azure.core.polling.AsyncLROPoller[~azure.mgmt.machinelearningservices.models.OutboundRuleBasicResource]
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        api_version = kwargs.pop('api_version', '2022-10-01')
        content_type = kwargs.pop('content_type', 'application/json')
        polling = kwargs.pop('polling', True)
        cls = kwargs.pop('cls', None)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = await self._create_or_update_initial(resource_group_name=resource_group_name, workspace_name=workspace_name, rule_name=rule_name, body=body, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            response = pipeline_response.http_response
            deserialized = self._deserialize('OutboundRuleBasicResource', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method = AsyncARMPolling(lro_delay, lro_options={'final-state-via': 'location'}, **kwargs)
        elif polling is False:
            polling_method = AsyncNoPolling()
        else:
            polling_method = polling
        if cont_token:
            return AsyncLROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return AsyncLROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/outboundRules/{ruleName}'}