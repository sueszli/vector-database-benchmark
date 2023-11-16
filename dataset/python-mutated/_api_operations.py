from typing import Any, Callable, Dict, IO, Iterable, Optional, TypeVar, Union, cast, overload
import urllib.parse
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.polling import LROPoller, NoPolling, PollingMethod
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from azure.mgmt.core.polling.arm_polling import ARMPolling
from .. import models as _models
from .._serialization import Serializer
from .._vendor import ApiManagementClientMixinABC, _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_by_service_request(resource_group_name: str, service_name: str, subscription_id: str, *, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, tags: Optional[str]=None, expand_api_version_set: Optional[bool]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str')
    if top is not None:
        _params['$top'] = _SERIALIZER.query('top', top, 'int', minimum=1)
    if skip is not None:
        _params['$skip'] = _SERIALIZER.query('skip', skip, 'int', minimum=0)
    if tags is not None:
        _params['tags'] = _SERIALIZER.query('tags', tags, 'str')
    if expand_api_version_set is not None:
        _params['expandApiVersionSet'] = _SERIALIZER.query('expand_api_version_set', expand_api_version_set, 'bool')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_entity_tag_request(resource_group_name: str, service_name: str, api_id: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'apiId': _SERIALIZER.url('api_id', api_id, 'str', max_length=256, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, service_name: str, api_id: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'apiId': _SERIALIZER.url('api_id', api_id, 'str', max_length=256, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, service_name: str, api_id: str, subscription_id: str, *, if_match: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'apiId': _SERIALIZER.url('api_id', api_id, 'str', max_length=256, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, service_name: str, api_id: str, subscription_id: str, *, if_match: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'apiId': _SERIALIZER.url('api_id', api_id, 'str', max_length=256, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, service_name: str, api_id: str, subscription_id: str, *, if_match: str, delete_revisions: Optional[bool]=None, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'apiId': _SERIALIZER.url('api_id', api_id, 'str', max_length=256, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    if delete_revisions is not None:
        _params['deleteRevisions'] = _SERIALIZER.query('delete_revisions', delete_revisions, 'bool')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_by_tags_request(resource_group_name: str, service_name: str, subscription_id: str, *, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, include_not_tagged_apis: Optional[bool]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apisByTags')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str')
    if top is not None:
        _params['$top'] = _SERIALIZER.query('top', top, 'int', minimum=1)
    if skip is not None:
        _params['$skip'] = _SERIALIZER.query('skip', skip, 'int', minimum=0)
    if include_not_tagged_apis is not None:
        _params['includeNotTaggedApis'] = _SERIALIZER.query('include_not_tagged_apis', include_not_tagged_apis, 'bool')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class ApiOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.apimanagement.ApiManagementClient`'s
        :attr:`api` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list_by_service(self, resource_group_name: str, service_name: str, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, tags: Optional[str]=None, expand_api_version_set: Optional[bool]=None, **kwargs: Any) -> Iterable['_models.ApiContract']:
        if False:
            while True:
                i = 10
        'Lists all APIs of the API Management service instance.\n\n        .. seealso::\n           - https://docs.microsoft.com/en-us/azure/api-management/api-management-howto-create-apis\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param filter: |     Field     |     Usage     |     Supported operators     |     Supported\n         functions     |</br>|-------------|-------------|-------------|-------------|</br>| name |\n         filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith |</br>|\n         displayName | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith\n         |</br>| description | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith,\n         endswith |</br>| serviceUrl | filter | ge, le, eq, ne, gt, lt | substringof, contains,\n         startswith, endswith |</br>| path | filter | ge, le, eq, ne, gt, lt | substringof, contains,\n         startswith, endswith |</br>| isCurrent | filter | eq, ne |  |</br>. Default value is None.\n        :type filter: str\n        :param top: Number of records to return. Default value is None.\n        :type top: int\n        :param skip: Number of records to skip. Default value is None.\n        :type skip: int\n        :param tags: Include tags in the response. Default value is None.\n        :type tags: str\n        :param expand_api_version_set: Include full ApiVersionSet resource in response. Default value\n         is None.\n        :type expand_api_version_set: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ApiContract or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.apimanagement.models.ApiContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ApiCollection] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_by_service_request(resource_group_name=resource_group_name, service_name=service_name, subscription_id=self._config.subscription_id, filter=filter, top=top, skip=skip, tags=tags, expand_api_version_set=expand_api_version_set, api_version=api_version, template_url=self.list_by_service.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('ApiCollection', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_service.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis'}

    @distributed_trace
    def get_entity_tag(self, resource_group_name: str, service_name: str, api_id: str, **kwargs: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Gets the entity state (Etag) version of the API specified by its identifier.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: bool or the result of cls(response)\n        :rtype: bool\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_get_entity_tag_request(resource_group_name=resource_group_name, service_name=service_name, api_id=api_id, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_entity_tag.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        if cls:
            return cls(pipeline_response, None, response_headers)
        return 200 <= response.status_code <= 299
    get_entity_tag.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}'}

    @distributed_trace
    def get(self, resource_group_name: str, service_name: str, api_id: str, **kwargs: Any) -> _models.ApiContract:
        if False:
            print('Hello World!')
        'Gets the details of the API specified by its identifier.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ApiContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.ApiContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ApiContract] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, service_name=service_name, api_id=api_id, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        deserialized = self._deserialize('ApiContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}'}

    def _create_or_update_initial(self, resource_group_name: str, service_name: str, api_id: str, parameters: Union[_models.ApiCreateOrUpdateParameter, IO], if_match: Optional[str]=None, **kwargs: Any) -> Optional[_models.ApiContract]:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.ApiContract]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'ApiCreateOrUpdateParameter')
        request = build_create_or_update_request(resource_group_name=resource_group_name, service_name=service_name, api_id=api_id, subscription_id=self._config.subscription_id, if_match=if_match, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_or_update_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('ApiContract', pipeline_response)
        if response.status_code == 201:
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('ApiContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_or_update_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}'}

    @overload
    def begin_create_or_update(self, resource_group_name: str, service_name: str, api_id: str, parameters: _models.ApiCreateOrUpdateParameter, if_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ApiContract]:
        if False:
            return 10
        'Creates new or updates existing specified API of the API Management service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :param parameters: Create or update parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.ApiCreateOrUpdateParameter\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ApiContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.ApiContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_or_update(self, resource_group_name: str, service_name: str, api_id: str, parameters: IO, if_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ApiContract]:
        if False:
            return 10
        'Creates new or updates existing specified API of the API Management service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :param parameters: Create or update parameters. Required.\n        :type parameters: IO\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ApiContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.ApiContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_or_update(self, resource_group_name: str, service_name: str, api_id: str, parameters: Union[_models.ApiCreateOrUpdateParameter, IO], if_match: Optional[str]=None, **kwargs: Any) -> LROPoller[_models.ApiContract]:
        if False:
            i = 10
            return i + 15
        "Creates new or updates existing specified API of the API Management service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :param parameters: Create or update parameters. Is either a ApiCreateOrUpdateParameter type or\n         a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.ApiCreateOrUpdateParameter or IO\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ApiContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.ApiContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ApiContract] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_or_update_initial(resource_group_name=resource_group_name, service_name=service_name, api_id=api_id, parameters=parameters, if_match=if_match, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            response_headers = {}
            response = pipeline_response.http_response
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('ApiContract', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, response_headers)
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'location'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}'}

    @overload
    def update(self, resource_group_name: str, service_name: str, api_id: str, if_match: str, parameters: _models.ApiUpdateContract, *, content_type: str='application/json', **kwargs: Any) -> _models.ApiContract:
        if False:
            return 10
        'Updates the specified API of the API Management service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :param parameters: API Update Contract parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.ApiUpdateContract\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ApiContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.ApiContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, service_name: str, api_id: str, if_match: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.ApiContract:
        if False:
            i = 10
            return i + 15
        'Updates the specified API of the API Management service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :param parameters: API Update Contract parameters. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ApiContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.ApiContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, service_name: str, api_id: str, if_match: str, parameters: Union[_models.ApiUpdateContract, IO], **kwargs: Any) -> _models.ApiContract:
        if False:
            print('Hello World!')
        "Updates the specified API of the API Management service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :param parameters: API Update Contract parameters. Is either a ApiUpdateContract type or a IO\n         type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.ApiUpdateContract or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ApiContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.ApiContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ApiContract] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'ApiUpdateContract')
        request = build_update_request(resource_group_name=resource_group_name, service_name=service_name, api_id=api_id, subscription_id=self._config.subscription_id, if_match=if_match, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        deserialized = self._deserialize('ApiContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}'}

    @distributed_trace
    def delete(self, resource_group_name: str, service_name: str, api_id: str, if_match: str, delete_revisions: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Deletes the specified API of the API Management service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param api_id: API revision identifier. Must be unique in the current API Management service\n         instance. Non-current revision has ;rev=n as a suffix where n is the revision number. Required.\n        :type api_id: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :param delete_revisions: Delete all revisions of the Api. Default value is None.\n        :type delete_revisions: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, service_name=service_name, api_id=api_id, subscription_id=self._config.subscription_id, if_match=if_match, delete_revisions=delete_revisions, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apis/{apiId}'}

    @distributed_trace
    def list_by_tags(self, resource_group_name: str, service_name: str, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, include_not_tagged_apis: Optional[bool]=None, **kwargs: Any) -> Iterable['_models.TagResourceContract']:
        if False:
            print('Hello World!')
        'Lists a collection of apis associated with tags.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param filter: |     Field     |     Usage     |     Supported operators     |     Supported\n         functions     |</br>|-------------|-------------|-------------|-------------|</br>| name |\n         filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith |</br>|\n         displayName | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith\n         |</br>| apiRevision | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith,\n         endswith |</br>| path | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith,\n         endswith |</br>| description | filter | ge, le, eq, ne, gt, lt | substringof, contains,\n         startswith, endswith |</br>| serviceUrl | filter | ge, le, eq, ne, gt, lt | substringof,\n         contains, startswith, endswith |</br>| isCurrent | filter | eq |     |</br>. Default value is\n         None.\n        :type filter: str\n        :param top: Number of records to return. Default value is None.\n        :type top: int\n        :param skip: Number of records to skip. Default value is None.\n        :type skip: int\n        :param include_not_tagged_apis: Include not tagged APIs. Default value is None.\n        :type include_not_tagged_apis: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either TagResourceContract or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.apimanagement.models.TagResourceContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.TagResourceCollection] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_by_tags_request(resource_group_name=resource_group_name, service_name=service_name, subscription_id=self._config.subscription_id, filter=filter, top=top, skip=skip, include_not_tagged_apis=include_not_tagged_apis, api_version=api_version, template_url=self.list_by_tags.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('TagResourceCollection', pipeline_response)
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
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_tags.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/apisByTags'}