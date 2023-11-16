import datetime
import sys
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar, Union
import urllib.parse
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from .._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_request(subscription_id: str, *, reported_start_time: datetime.datetime, reported_end_time: datetime.datetime, show_details: Optional[bool]=None, aggregation_granularity: Union[str, _models.AggregationGranularity]='Daily', continuation_token_parameter: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2015-06-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2015-06-01-preview'))
    accept = _headers.pop('Accept', 'application/json, text/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/providers/Microsoft.Commerce/UsageAggregates')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['reportedStartTime'] = _SERIALIZER.query('reported_start_time', reported_start_time, 'iso-8601')
    _params['reportedEndTime'] = _SERIALIZER.query('reported_end_time', reported_end_time, 'iso-8601')
    if show_details is not None:
        _params['showDetails'] = _SERIALIZER.query('show_details', show_details, 'bool')
    if aggregation_granularity is not None:
        _params['aggregationGranularity'] = _SERIALIZER.query('aggregation_granularity', aggregation_granularity, 'str')
    if continuation_token_parameter is not None:
        _params['continuationToken'] = _SERIALIZER.query('continuation_token_parameter', continuation_token_parameter, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class UsageAggregatesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.commerce.UsageManagementClient`'s
        :attr:`usage_aggregates` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list(self, reported_start_time: datetime.datetime, reported_end_time: datetime.datetime, show_details: Optional[bool]=None, aggregation_granularity: Union[str, _models.AggregationGranularity]='Daily', continuation_token_parameter: Optional[str]=None, **kwargs: Any) -> Iterable['_models.UsageAggregation']:
        if False:
            print('Hello World!')
        'Query aggregated Azure subscription consumption data for a date range.\n\n        .. seealso::\n           - https://docs.microsoft.com/rest/api/commerce/usageaggregates\n\n        :param reported_start_time: The start of the time range to retrieve data for. Required.\n        :type reported_start_time: ~datetime.datetime\n        :param reported_end_time: The end of the time range to retrieve data for. Required.\n        :type reported_end_time: ~datetime.datetime\n        :param show_details: ``True`` returns usage data in instance-level detail, ``false`` causes\n         server-side aggregation with fewer details. For example, if you have 3 website instances, by\n         default you will get 3 line items for website consumption. If you specify showDetails = false,\n         the data will be aggregated as a single line item for website consumption within the time\n         period (for the given subscriptionId, meterId, usageStartTime and usageEndTime). Default value\n         is None.\n        :type show_details: bool\n        :param aggregation_granularity: ``Daily`` (default) returns the data in daily granularity,\n         ``Hourly`` returns the data in hourly granularity. Known values are: "Daily" and "Hourly".\n         Default value is "Daily".\n        :type aggregation_granularity: str or ~azure.mgmt.commerce.models.AggregationGranularity\n        :param continuation_token_parameter: Used when a continuation token string is provided in the\n         response body of the previous call, enabling paging through a large result set. If not present,\n         the data is retrieved from the beginning of the day/hour (based on the granularity) passed in.\n         Default value is None.\n        :type continuation_token_parameter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either UsageAggregation or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.commerce.models.UsageAggregation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2015-06-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.UsageAggregationListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_request(subscription_id=self._config.subscription_id, reported_start_time=reported_start_time, reported_end_time=reported_end_time, show_details=show_details, aggregation_granularity=aggregation_granularity, continuation_token_parameter=continuation_token_parameter, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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

        def extract_data(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('UsageAggregationListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.Commerce/UsageAggregates'}