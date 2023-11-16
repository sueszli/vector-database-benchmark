from typing import Any, AsyncIterable, Callable, Dict, Optional, TypeVar, Union
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
from ...operations._benefit_utilization_summaries_operations import build_list_by_billing_account_id_request, build_list_by_billing_profile_id_request, build_list_by_savings_plan_id_request, build_list_by_savings_plan_order_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class BenefitUtilizationSummariesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.costmanagement.aio.CostManagementClient`'s
        :attr:`benefit_utilization_summaries` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list_by_billing_account_id(self, billing_account_id: str, grain_parameter: Optional[Union[str, _models.GrainParameter]]=None, filter: Optional[str]=None, **kwargs: Any) -> AsyncIterable['_models.BenefitUtilizationSummary']:
        if False:
            for i in range(10):
                print('nop')
        'Lists savings plan utilization summaries for the enterprise agreement scope. Supported at grain\n        values: \'Daily\' and \'Monthly\'.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/rest/api/cost-management/\n\n        :param billing_account_id: Billing account ID. Required.\n        :type billing_account_id: str\n        :param grain_parameter: Grain. Known values are: "Hourly", "Daily", and "Monthly". Default\n         value is None.\n        :type grain_parameter: str or ~azure.mgmt.costmanagement.models.GrainParameter\n        :param filter: Supports filtering by properties/benefitId, properties/benefitOrderId and\n         properties/usageDate. Default value is None.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BenefitUtilizationSummary or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.costmanagement.models.BenefitUtilizationSummary]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.BenefitUtilizationSummariesListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_by_billing_account_id_request(billing_account_id=billing_account_id, grain_parameter=grain_parameter, filter=filter, api_version=api_version, template_url=self.list_by_billing_account_id.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('BenefitUtilizationSummariesListResult', pipeline_response)
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
    list_by_billing_account_id.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/providers/Microsoft.CostManagement/benefitUtilizationSummaries'}

    @distributed_trace
    def list_by_billing_profile_id(self, billing_account_id: str, billing_profile_id: str, grain_parameter: Optional[Union[str, _models.GrainParameter]]=None, filter: Optional[str]=None, **kwargs: Any) -> AsyncIterable['_models.BenefitUtilizationSummary']:
        if False:
            i = 10
            return i + 15
        'Lists savings plan utilization summaries for billing profile. Supported at grain values:\n        \'Daily\' and \'Monthly\'.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/rest/api/cost-management/\n\n        :param billing_account_id: Billing account ID. Required.\n        :type billing_account_id: str\n        :param billing_profile_id: Billing profile ID. Required.\n        :type billing_profile_id: str\n        :param grain_parameter: Grain. Known values are: "Hourly", "Daily", and "Monthly". Default\n         value is None.\n        :type grain_parameter: str or ~azure.mgmt.costmanagement.models.GrainParameter\n        :param filter: Supports filtering by properties/benefitId, properties/benefitOrderId and\n         properties/usageDate. Default value is None.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BenefitUtilizationSummary or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.costmanagement.models.BenefitUtilizationSummary]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.BenefitUtilizationSummariesListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_by_billing_profile_id_request(billing_account_id=billing_account_id, billing_profile_id=billing_profile_id, grain_parameter=grain_parameter, filter=filter, api_version=api_version, template_url=self.list_by_billing_profile_id.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('BenefitUtilizationSummariesListResult', pipeline_response)
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
    list_by_billing_profile_id.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}/providers/Microsoft.CostManagement/benefitUtilizationSummaries'}

    @distributed_trace
    def list_by_savings_plan_order(self, savings_plan_order_id: str, filter: Optional[str]=None, grain_parameter: Optional[Union[str, _models.GrainParameter]]=None, **kwargs: Any) -> AsyncIterable['_models.BenefitUtilizationSummary']:
        if False:
            return 10
        'Lists the savings plan utilization summaries for daily or monthly grain.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/rest/api/cost-management/\n\n        :param savings_plan_order_id: Savings plan order ID. Required.\n        :type savings_plan_order_id: str\n        :param filter: Supports filtering by properties/usageDate. Default value is None.\n        :type filter: str\n        :param grain_parameter: Grain. Known values are: "Hourly", "Daily", and "Monthly". Default\n         value is None.\n        :type grain_parameter: str or ~azure.mgmt.costmanagement.models.GrainParameter\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BenefitUtilizationSummary or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.costmanagement.models.BenefitUtilizationSummary]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.BenefitUtilizationSummariesListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_by_savings_plan_order_request(savings_plan_order_id=savings_plan_order_id, filter=filter, grain_parameter=grain_parameter, api_version=api_version, template_url=self.list_by_savings_plan_order.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('BenefitUtilizationSummariesListResult', pipeline_response)
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
    list_by_savings_plan_order.metadata = {'url': '/providers/Microsoft.BillingBenefits/savingsPlanOrders/{savingsPlanOrderId}/providers/Microsoft.CostManagement/benefitUtilizationSummaries'}

    @distributed_trace
    def list_by_savings_plan_id(self, savings_plan_order_id: str, savings_plan_id: str, filter: Optional[str]=None, grain_parameter: Optional[Union[str, _models.GrainParameter]]=None, **kwargs: Any) -> AsyncIterable['_models.BenefitUtilizationSummary']:
        if False:
            return 10
        'Lists the savings plan utilization summaries for daily or monthly grain.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/rest/api/cost-management/\n\n        :param savings_plan_order_id: Savings plan order ID. Required.\n        :type savings_plan_order_id: str\n        :param savings_plan_id: Savings plan ID. Required.\n        :type savings_plan_id: str\n        :param filter: Supports filtering by properties/usageDate. Default value is None.\n        :type filter: str\n        :param grain_parameter: Grain. Known values are: "Hourly", "Daily", and "Monthly". Default\n         value is None.\n        :type grain_parameter: str or ~azure.mgmt.costmanagement.models.GrainParameter\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BenefitUtilizationSummary or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.costmanagement.models.BenefitUtilizationSummary]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.BenefitUtilizationSummariesListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_by_savings_plan_id_request(savings_plan_order_id=savings_plan_order_id, savings_plan_id=savings_plan_id, filter=filter, grain_parameter=grain_parameter, api_version=api_version, template_url=self.list_by_savings_plan_id.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('BenefitUtilizationSummariesListResult', pipeline_response)
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
    list_by_savings_plan_id.metadata = {'url': '/providers/Microsoft.BillingBenefits/savingsPlanOrders/{savingsPlanOrderId}/savingsPlans/{savingsPlanId}/providers/Microsoft.CostManagement/benefitUtilizationSummaries'}