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
from ...operations._dimensions_operations import build_by_external_cloud_provider_type_request, build_list_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class DimensionsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.costmanagement.aio.CostManagementClient`'s
        :attr:`dimensions` attribute.
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
    def list(self, scope: str, filter: Optional[str]=None, expand: Optional[str]=None, skiptoken: Optional[str]=None, top: Optional[int]=None, **kwargs: Any) -> AsyncIterable['_models.Dimension']:
        if False:
            return 10
        "Lists the dimensions by the defined scope.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/rest/api/costmanagement/\n\n        :param scope: The scope associated with dimension operations. This includes\n         '/subscriptions/{subscriptionId}/' for subscription scope,\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}' for resourceGroup scope,\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}' for Billing Account scope,\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/departments/{departmentId}'\n         for Department scope,\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/enrollmentAccounts/{enrollmentAccountId}'\n         for EnrollmentAccount scope,\n         '/providers/Microsoft.Management/managementGroups/{managementGroupId}' for Management Group\n         scope,\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}'\n         for billingProfile scope,\n         'providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}/invoiceSections/{invoiceSectionId}'\n         for invoiceSection scope, and\n         'providers/Microsoft.Billing/billingAccounts/{billingAccountId}/customers/{customerId}'\n         specific for partners. Required.\n        :type scope: str\n        :param filter: May be used to filter dimensions by properties/category, properties/usageStart,\n         properties/usageEnd. Supported operators are 'eq','lt', 'gt', 'le', 'ge'. Default value is\n         None.\n        :type filter: str\n        :param expand: May be used to expand the properties/data within a dimension category. By\n         default, data is not included when listing dimensions. Default value is None.\n        :type expand: str\n        :param skiptoken: Skiptoken is only used if a previous operation returned a partial result. If\n         a previous response contains a nextLink element, the value of the nextLink element will include\n         a skiptoken parameter that specifies a starting point to use for subsequent calls. Default\n         value is None.\n        :type skiptoken: str\n        :param top: May be used to limit the number of results to the most recent N dimension data.\n         Default value is None.\n        :type top: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Dimension or the result of cls(response)\n        :rtype: ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.costmanagement.models.Dimension]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.DimensionsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_request(scope=scope, filter=filter, expand=expand, skiptoken=skiptoken, top=top, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('DimensionsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, AsyncList(list_of_elem))

        async def get_next(next_link=None):
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = await self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200, 204]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return AsyncItemPaged(get_next, extract_data)
    list.metadata = {'url': '/{scope}/providers/Microsoft.CostManagement/dimensions'}

    @distributed_trace
    def by_external_cloud_provider_type(self, external_cloud_provider_type: Union[str, _models.ExternalCloudProviderType], external_cloud_provider_id: str, filter: Optional[str]=None, expand: Optional[str]=None, skiptoken: Optional[str]=None, top: Optional[int]=None, **kwargs: Any) -> AsyncIterable['_models.Dimension']:
        if False:
            while True:
                i = 10
        'Lists the dimensions by the external cloud provider type.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/rest/api/costmanagement/\n\n        :param external_cloud_provider_type: The external cloud provider type associated with\n         dimension/query operations. This includes \'externalSubscriptions\' for linked account and\n         \'externalBillingAccounts\' for consolidated account. Known values are: "externalSubscriptions"\n         and "externalBillingAccounts". Required.\n        :type external_cloud_provider_type: str or\n         ~azure.mgmt.costmanagement.models.ExternalCloudProviderType\n        :param external_cloud_provider_id: This can be \'{externalSubscriptionId}\' for linked account or\n         \'{externalBillingAccountId}\' for consolidated account used with dimension/query operations.\n         Required.\n        :type external_cloud_provider_id: str\n        :param filter: May be used to filter dimensions by properties/category, properties/usageStart,\n         properties/usageEnd. Supported operators are \'eq\',\'lt\', \'gt\', \'le\', \'ge\'. Default value is\n         None.\n        :type filter: str\n        :param expand: May be used to expand the properties/data within a dimension category. By\n         default, data is not included when listing dimensions. Default value is None.\n        :type expand: str\n        :param skiptoken: Skiptoken is only used if a previous operation returned a partial result. If\n         a previous response contains a nextLink element, the value of the nextLink element will include\n         a skiptoken parameter that specifies a starting point to use for subsequent calls. Default\n         value is None.\n        :type skiptoken: str\n        :param top: May be used to limit the number of results to the most recent N dimension data.\n         Default value is None.\n        :type top: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Dimension or the result of cls(response)\n        :rtype: ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.costmanagement.models.Dimension]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.DimensionsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_by_external_cloud_provider_type_request(external_cloud_provider_type=external_cloud_provider_type, external_cloud_provider_id=external_cloud_provider_id, filter=filter, expand=expand, skiptoken=skiptoken, top=top, api_version=api_version, template_url=self.by_external_cloud_provider_type.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('DimensionsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, AsyncList(list_of_elem))

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
    by_external_cloud_provider_type.metadata = {'url': '/providers/Microsoft.CostManagement/{externalCloudProviderType}/{externalCloudProviderId}/dimensions'}