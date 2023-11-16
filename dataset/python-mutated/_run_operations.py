from typing import Any, AsyncIterable, Callable, Dict, List, Optional, TypeVar, Union
from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import AsyncHttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.mgmt.core.exceptions import ARMErrorFormat
from ... import models as _models
from ..._vendor import _convert_request
from ...operations._run_operations import build_list_by_compute_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class RunOperations:
    """RunOperations async operations.

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
            while True:
                i = 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def list_by_compute(self, subscription_id: str, resource_group_name: str, workspace_name: str, compute_name: str, filter: Optional[str]=None, continuationtoken: Optional[str]=None, orderby: Optional[List[str]]=None, sortorder: Optional[Union[str, '_models.SortOrderDirection']]=None, top: Optional[int]=None, count: Optional[bool]=None, **kwargs: Any) -> AsyncIterable['_models.PaginatedRunList']:
        if False:
            return 10
        'list_by_compute.\n\n        :param subscription_id: The Azure Subscription ID.\n        :type subscription_id: str\n        :param resource_group_name: The Name of the resource group in which the workspace is located.\n        :type resource_group_name: str\n        :param workspace_name: The name of the workspace.\n        :type workspace_name: str\n        :param compute_name:\n        :type compute_name: str\n        :param filter: Allows for filtering the collection of resources.\n         The expression specified is evaluated for each resource in the collection, and only items\n         where the expression evaluates to true are included in the response.\n        :type filter: str\n        :param continuationtoken: The continuation token to use for getting the next set of resources.\n        :type continuationtoken: str\n        :param orderby: The list of resource properties to use for sorting the requested resources.\n        :type orderby: list[str]\n        :param sortorder: The sort order of the returned resources. Not used, specify asc or desc after\n         each property name in the OrderBy parameter.\n        :type sortorder: str or ~azure.mgmt.machinelearningservices.models.SortOrderDirection\n        :param top: The maximum number of items in the resource collection to be included in the\n         result.\n         If not specified, all items are returned.\n        :type top: int\n        :param count: Whether to include a count of the matching resources along with the resources\n         returned in the response.\n        :type count: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either PaginatedRunList or the result of cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.machinelearningservices.models.PaginatedRunList]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_by_compute_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, compute_name=compute_name, filter=filter, continuationtoken=continuationtoken, orderby=orderby, sortorder=sortorder, top=top, count=count, template_url=self.list_by_compute.metadata['url'])
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                request = build_list_by_compute_request(subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, compute_name=compute_name, filter=filter, continuationtoken=continuationtoken, orderby=orderby, sortorder=sortorder, top=top, count=count, template_url=next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize('PaginatedRunList', pipeline_response)
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
    list_by_compute.metadata = {'url': '/history/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/computes/{computeName}/runs'}