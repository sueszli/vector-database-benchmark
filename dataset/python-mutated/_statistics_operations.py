import sys
from typing import Any, AsyncIterable, Callable, Dict, Optional, TypeVar
from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import AsyncHttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from ... import models as _models
from ..._vendor import _convert_request
from ...operations._statistics_operations import build_list_by_automation_account_request
from .._vendor import AutomationClientMixinABC
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class StatisticsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.automation.aio.AutomationClient`'s
        :attr:`statistics` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list_by_automation_account(self, resource_group_name: str, automation_account_name: str, filter: Optional[str]=None, **kwargs: Any) -> AsyncIterable['_models.Statistics']:
        if False:
            while True:
                i = 10
        'Retrieve the statistics for the account.\n\n        :param resource_group_name: Name of an Azure Resource group. Required.\n        :type resource_group_name: str\n        :param automation_account_name: The name of the automation account. Required.\n        :type automation_account_name: str\n        :param filter: The filter to apply on the operation. Default value is None.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Statistics or the result of cls(response)\n        :rtype: ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.automation.models.Statistics]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-08-08'] = kwargs.pop('api_version', _params.pop('api-version', '2022-08-08'))
        cls: ClsType[_models.StatisticsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_by_automation_account_request(resource_group_name=resource_group_name, automation_account_name=automation_account_name, subscription_id=self._config.subscription_id, filter=filter, api_version=api_version, template_url=self.list_by_automation_account.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                request = HttpRequest('GET', next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize('StatisticsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, AsyncList(list_of_elem))

        async def get_next(next_link=None):
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = await self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return AsyncItemPaged(get_next, extract_data)
    list_by_automation_account.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Automation/automationAccounts/{automationAccountName}/statistics'}