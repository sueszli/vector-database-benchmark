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
from ...operations._reservations_details_operations import build_list_by_reservation_order_and_reservation_request, build_list_by_reservation_order_request, build_list_request
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class ReservationsDetailsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.consumption.aio.ConsumptionManagementClient`'s
        :attr:`reservations_details` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list_by_reservation_order(self, reservation_order_id: str, filter: str, **kwargs: Any) -> AsyncIterable['_models.ReservationDetail']:
        if False:
            while True:
                i = 10
        "Lists the reservations details for provided date range. Note: ARM has a payload size limit of\n        12MB, so currently callers get 502 when the response size exceeds the ARM limit. In such cases,\n        API call should be made with smaller date ranges.\n\n        :param reservation_order_id: Order Id of the reservation. Required.\n        :type reservation_order_id: str\n        :param filter: Filter reservation details by date range. The properties/UsageDate for start\n         date and end date. The filter supports 'le' and  'ge'. Required.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ReservationDetail or the result of cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.consumption.models.ReservationDetail]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2021-10-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ReservationDetailsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                request = build_list_by_reservation_order_request(reservation_order_id=reservation_order_id, filter=filter, api_version=api_version, template_url=self.list_by_reservation_order.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('ReservationDetailsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, AsyncList(list_of_elem))

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
    list_by_reservation_order.metadata = {'url': '/providers/Microsoft.Capacity/reservationorders/{reservationOrderId}/providers/Microsoft.Consumption/reservationDetails'}

    @distributed_trace
    def list_by_reservation_order_and_reservation(self, reservation_order_id: str, reservation_id: str, filter: str, **kwargs: Any) -> AsyncIterable['_models.ReservationDetail']:
        if False:
            print('Hello World!')
        "Lists the reservations details for provided date range. Note: ARM has a payload size limit of\n        12MB, so currently callers get 502 when the response size exceeds the ARM limit. In such cases,\n        API call should be made with smaller date ranges.\n\n        :param reservation_order_id: Order Id of the reservation. Required.\n        :type reservation_order_id: str\n        :param reservation_id: Id of the reservation. Required.\n        :type reservation_id: str\n        :param filter: Filter reservation details by date range. The properties/UsageDate for start\n         date and end date. The filter supports 'le' and  'ge'. Required.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ReservationDetail or the result of cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.consumption.models.ReservationDetail]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2021-10-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ReservationDetailsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_by_reservation_order_and_reservation_request(reservation_order_id=reservation_order_id, reservation_id=reservation_id, filter=filter, api_version=api_version, template_url=self.list_by_reservation_order_and_reservation.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('ReservationDetailsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, AsyncList(list_of_elem))

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
    list_by_reservation_order_and_reservation.metadata = {'url': '/providers/Microsoft.Capacity/reservationorders/{reservationOrderId}/reservations/{reservationId}/providers/Microsoft.Consumption/reservationDetails'}

    @distributed_trace
    def list(self, resource_scope: str, start_date: Optional[str]=None, end_date: Optional[str]=None, filter: Optional[str]=None, reservation_id: Optional[str]=None, reservation_order_id: Optional[str]=None, **kwargs: Any) -> AsyncIterable['_models.ReservationDetail']:
        if False:
            i = 10
            return i + 15
        "Lists the reservations details for the defined scope and provided date range. Note: ARM has a\n        payload size limit of 12MB, so currently callers get 502 when the response size exceeds the ARM\n        limit. In such cases, API call should be made with smaller date ranges.\n\n        :param resource_scope: The scope associated with reservations details operations. This includes\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}' for BillingAccount scope\n         (legacy), and\n         '/providers/Microsoft.Billing/billingAccounts/{billingAccountId}/billingProfiles/{billingProfileId}'\n         for BillingProfile scope (modern). Required.\n        :type resource_scope: str\n        :param start_date: Start date. Only applicable when querying with billing profile. Default\n         value is None.\n        :type start_date: str\n        :param end_date: End date. Only applicable when querying with billing profile. Default value is\n         None.\n        :type end_date: str\n        :param filter: Filter reservation details by date range. The properties/UsageDate for start\n         date and end date. The filter supports 'le' and  'ge'. Not applicable when querying with\n         billing profile. Default value is None.\n        :type filter: str\n        :param reservation_id: Reservation Id GUID. Only valid if reservationOrderId is also provided.\n         Filter to a specific reservation. Default value is None.\n        :type reservation_id: str\n        :param reservation_order_id: Reservation Order Id GUID. Required if reservationId is provided.\n         Filter to a specific reservation order. Default value is None.\n        :type reservation_order_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ReservationDetail or the result of cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.consumption.models.ReservationDetail]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2021-10-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ReservationDetailsListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_request(resource_scope=resource_scope, start_date=start_date, end_date=end_date, filter=filter, reservation_id=reservation_id, reservation_order_id=reservation_order_id, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('ReservationDetailsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, AsyncList(list_of_elem))

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
    list.metadata = {'url': '/{resourceScope}/providers/Microsoft.Consumption/reservationDetails'}