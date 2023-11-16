import sys
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
from ...operations._usage_details_operations import build_list_request
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class UsageDetailsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.consumption.aio.ConsumptionManagementClient`'s
        :attr:`usage_details` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list(self, scope: str, expand: Optional[str]=None, filter: Optional[str]=None, skiptoken: Optional[str]=None, top: Optional[int]=None, metric: Optional[Union[str, _models.Metrictype]]=None, **kwargs: Any) -> AsyncIterable['_models.UsageDetail']:
        if False:
            print('Hello World!')
        'Lists the usage details for the defined scope. Usage details are available via this API only\n        for May 1, 2014 or later.\n\n        :param scope: The scope associated with usage details operations. This includes\n         \'/subscriptions/{subscriptionId}/\' for subscription scope,\n         \'/providers/Microsoft.Billing/billingAccounts/{billingAccountId}\' for Billing Account scope,\n         \'/providers/Microsoft.Billing/departments/{departmentId}\' for Department scope,\n         \'/providers/Microsoft.Billing/enrollmentAccounts/{enrollmentAccountId}\' for EnrollmentAccount\n         scope and \'/providers/Microsoft.Management/managementGroups/{managementGroupId}\' for Management\n         Group scope. For subscription, billing account, department, enrollment account and management\n         group, you can also add billing period to the scope using\n         \'/providers/Microsoft.Billing/billingPeriods/{billingPeriodName}\'. For e.g. to specify billing\n         period at department scope use\n         \'/providers/Microsoft.Billing/departments/{departmentId}/providers/Microsoft.Billing/billingPeriods/{billingPeriodName}\'.\n         Also, Modern Commerce Account scopes are\n         \'/providers/Microsoft.Billing/billingAccounts/{billingAccountId}\' for billingAccount scope,\n         \'/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}\'\n         for billingProfile scope,\n         \'providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}/invoiceSections/{invoiceSectionId}\'\n         for invoiceSection scope, and\n         \'providers/Microsoft.Billing/billingAccounts/{billingAccountId}/customers/{customerId}\'\n         specific for partners. Required.\n        :type scope: str\n        :param expand: May be used to expand the properties/additionalInfo or properties/meterDetails\n         within a list of usage details. By default, these fields are not included when listing usage\n         details. Default value is None.\n        :type expand: str\n        :param filter: May be used to filter usageDetails by properties/resourceGroup,\n         properties/resourceName, properties/resourceId, properties/chargeType,\n         properties/reservationId, properties/publisherType or tags. The filter supports \'eq\', \'lt\',\n         \'gt\', \'le\', \'ge\', and \'and\'. It does not currently support \'ne\', \'or\', or \'not\'. Tag filter is\n         a key value pair string where key and value is separated by a colon (:). PublisherType Filter\n         accepts two values azure and marketplace and it is currently supported for Web Direct Offer\n         Type. Default value is None.\n        :type filter: str\n        :param skiptoken: Skiptoken is only used if a previous operation returned a partial result. If\n         a previous response contains a nextLink element, the value of the nextLink element will include\n         a skiptoken parameter that specifies a starting point to use for subsequent calls. Default\n         value is None.\n        :type skiptoken: str\n        :param top: May be used to limit the number of results to the most recent N usageDetails.\n         Default value is None.\n        :type top: int\n        :param metric: Allows to select different type of cost/usage records. Known values are:\n         "actualcost", "amortizedcost", and "usage". Default value is None.\n        :type metric: str or ~azure.mgmt.consumption.models.Metrictype\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either UsageDetail or the result of cls(response)\n        :rtype: ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.consumption.models.UsageDetail]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2021-10-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.UsageDetailsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_request(scope=scope, expand=expand, filter=filter, skiptoken=skiptoken, top=top, metric=metric, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('UsageDetailsListResult', pipeline_response)
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
    list.metadata = {'url': '/{scope}/providers/Microsoft.Consumption/usageDetails'}