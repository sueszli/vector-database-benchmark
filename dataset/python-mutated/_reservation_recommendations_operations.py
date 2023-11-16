import sys
from typing import Any, AsyncIterable, Callable, Dict, Optional, TypeVar
import urllib.parse
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
from ...operations._reservation_recommendations_operations import build_list_request
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class ReservationRecommendationsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.consumption.aio.ConsumptionManagementClient`'s
        :attr:`reservation_recommendations` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list(self, resource_scope: str, filter: Optional[str]=None, **kwargs: Any) -> AsyncIterable['_models.ReservationRecommendation']:
        if False:
            i = 10
            return i + 15
        "List of recommendations for purchasing reserved instances.\n\n        :param resource_scope: The scope associated with reservation recommendations operations. This\n         includes '/subscriptions/{subscriptionId}/' for subscription scope,\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}' for resource group scope,\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}' for BillingAccount scope, and\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}'\n         for billingProfile scope. Required.\n        :type resource_scope: str\n        :param filter: May be used to filter reservationRecommendations by: properties/scope with\n         allowed values ['Single', 'Shared'] and default value 'Single'; properties/resourceType with\n         allowed values ['VirtualMachines', 'SQLDatabases', 'PostgreSQL', 'ManagedDisk', 'MySQL',\n         'RedHat', 'MariaDB', 'RedisCache', 'CosmosDB', 'SqlDataWarehouse', 'SUSELinux', 'AppService',\n         'BlockBlob', 'AzureDataExplorer', 'VMwareCloudSimple'] and default value 'VirtualMachines'; and\n         properties/lookBackPeriod with allowed values ['Last7Days', 'Last30Days', 'Last60Days'] and\n         default value 'Last7Days'. Default value is None.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ReservationRecommendation or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.consumption.models.ReservationRecommendation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2021-10-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ReservationRecommendationsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_request(resource_scope=resource_scope, filter=filter, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize('ReservationRecommendationsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, AsyncList(list_of_elem))

        async def get_next(next_link=None):
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = await self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200, 204]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return AsyncItemPaged(get_next, extract_data)
    list.metadata = {'url': '/{resourceScope}/providers/Microsoft.Consumption/reservationRecommendations'}