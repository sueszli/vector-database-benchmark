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
from ...operations._benefit_recommendations_operations import build_list_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class BenefitRecommendationsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.costmanagement.aio.CostManagementClient`'s
        :attr:`benefit_recommendations` attribute.
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
    def list(self, billing_scope: str, filter: Optional[str]=None, orderby: Optional[str]=None, expand: Optional[str]=None, **kwargs: Any) -> AsyncIterable['_models.BenefitRecommendationModel']:
        if False:
            while True:
                i = 10
        "List of recommendations for purchasing savings plan.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/rest/api/CostManagement/\n\n        :param billing_scope: The scope associated with benefit recommendation operations. This\n         includes '/subscriptions/{subscriptionId}/' for subscription scope,\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}' for resource group scope,\n         /providers/Microsoft.Billing/billingAccounts/{billingAccountId}' for enterprise agreement\n         scope, and\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}'\n         for billing profile scope. Required.\n        :type billing_scope: str\n        :param filter: Can be used to filter benefitRecommendations by: properties/scope with allowed\n         values ['Single', 'Shared'] and default value 'Shared'; and properties/lookBackPeriod with\n         allowed values ['Last7Days', 'Last30Days', 'Last60Days'] and default value 'Last60Days';\n         properties/term with allowed values ['P1Y', 'P3Y'] and default value 'P3Y';\n         properties/subscriptionId; properties/resourceGroup. Default value is None.\n        :type filter: str\n        :param orderby: May be used to order the recommendations by: properties/armSkuName. For the\n         savings plan, the results are in order by default. There is no need to use this clause. Default\n         value is None.\n        :type orderby: str\n        :param expand: May be used to expand the properties by: properties/usage,\n         properties/allRecommendationDetails. Default value is None.\n        :type expand: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BenefitRecommendationModel or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.costmanagement.models.BenefitRecommendationModel]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.BenefitRecommendationsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_request(billing_scope=billing_scope, filter=filter, orderby=orderby, expand=expand, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('BenefitRecommendationsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, AsyncList(list_of_elem))

        async def get_next(next_link=None):
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = await self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return AsyncItemPaged(get_next, extract_data)
    list.metadata = {'url': '/{billingScope}/providers/Microsoft.CostManagement/benefitRecommendations'}