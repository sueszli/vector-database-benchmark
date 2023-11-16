from typing import Any, Callable, Dict, Iterable, Optional, TypeVar
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
from ..._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(scope: str, role_eligibility_schedule_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2020-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleEligibilitySchedules/{roleEligibilityScheduleName}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'roleEligibilityScheduleName': _SERIALIZER.url('role_eligibility_schedule_name', role_eligibility_schedule_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_for_scope_request(scope: str, *, filter: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2020-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleEligibilitySchedules')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class RoleEligibilitySchedulesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.authorization.v2020_10_01.AuthorizationManagementClient`'s
        :attr:`role_eligibility_schedules` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')
        self._api_version = input_args.pop(0) if input_args else kwargs.pop('api_version')

    @distributed_trace
    def get(self, scope: str, role_eligibility_schedule_name: str, **kwargs: Any) -> _models.RoleEligibilitySchedule:
        if False:
            for i in range(10):
                print('nop')
        'Get the specified role eligibility schedule for a resource scope.\n\n        :param scope: The scope of the role eligibility schedule. Required.\n        :type scope: str\n        :param role_eligibility_schedule_name: The name (guid) of the role eligibility schedule to get.\n         Required.\n        :type role_eligibility_schedule_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleEligibilitySchedule or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2020_10_01.models.RoleEligibilitySchedule\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2020-10-01'))
        cls: ClsType[_models.RoleEligibilitySchedule] = kwargs.pop('cls', None)
        request = build_get_request(scope=scope, role_eligibility_schedule_name=role_eligibility_schedule_name, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('RoleEligibilitySchedule', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleEligibilitySchedules/{roleEligibilityScheduleName}'}

    @distributed_trace
    def list_for_scope(self, scope: str, filter: Optional[str]=None, **kwargs: Any) -> Iterable['_models.RoleEligibilitySchedule']:
        if False:
            while True:
                i = 10
        "Gets role eligibility schedules for a resource scope.\n\n        :param scope: The scope of the role eligibility schedules. Required.\n        :type scope: str\n        :param filter: The filter to apply on the operation. Use $filter=atScope() to return all role\n         eligibility schedules at or above the scope. Use $filter=principalId eq {id} to return all role\n         eligibility schedules at, above or below the scope for the specified principal. Use\n         $filter=assignedTo('{userId}') to return all role eligibility schedules for the user. Use\n         $filter=asTarget() to return all role eligibility schedules created for the current user.\n         Default value is None.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either RoleEligibilitySchedule or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.authorization.v2020_10_01.models.RoleEligibilitySchedule]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2020-10-01'))
        cls: ClsType[_models.RoleEligibilityScheduleListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_for_scope_request(scope=scope, filter=filter, api_version=api_version, template_url=self.list_for_scope.metadata['url'], headers=_headers, params=_params)
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
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('RoleEligibilityScheduleListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_for_scope.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleEligibilitySchedules'}