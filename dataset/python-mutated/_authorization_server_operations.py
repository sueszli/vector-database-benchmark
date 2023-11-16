from typing import Any, Callable, Dict, IO, Iterable, Optional, TypeVar, Union, overload
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
from .._vendor import ApiManagementClientMixinABC, _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_by_service_request(resource_group_name: str, service_name: str, subscription_id: str, *, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str')
    if top is not None:
        _params['$top'] = _SERIALIZER.query('top', top, 'int', minimum=1)
    if skip is not None:
        _params['$skip'] = _SERIALIZER.query('skip', skip, 'int', minimum=0)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_entity_tag_request(resource_group_name: str, service_name: str, authsid: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'authsid': _SERIALIZER.url('authsid', authsid, 'str', max_length=80, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, service_name: str, authsid: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'authsid': _SERIALIZER.url('authsid', authsid, 'str', max_length=80, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, service_name: str, authsid: str, subscription_id: str, *, if_match: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'authsid': _SERIALIZER.url('authsid', authsid, 'str', max_length=80, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, service_name: str, authsid: str, subscription_id: str, *, if_match: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'authsid': _SERIALIZER.url('authsid', authsid, 'str', max_length=80, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, service_name: str, authsid: str, subscription_id: str, *, if_match: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'authsid': _SERIALIZER.url('authsid', authsid, 'str', max_length=80, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_secrets_request(resource_group_name: str, service_name: str, authsid: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}/listSecrets')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'authsid': _SERIALIZER.url('authsid', authsid, 'str', max_length=80, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class AuthorizationServerOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.apimanagement.ApiManagementClient`'s
        :attr:`authorization_server` attribute.
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
    def list_by_service(self, resource_group_name: str, service_name: str, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, **kwargs: Any) -> Iterable['_models.AuthorizationServerContract']:
        if False:
            return 10
        'Lists a collection of authorization servers defined within a service instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param filter: |     Field     |     Usage     |     Supported operators     |     Supported\n         functions     |</br>|-------------|-------------|-------------|-------------|</br>| name |\n         filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith |</br>|\n         displayName | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith\n         |</br>. Default value is None.\n        :type filter: str\n        :param top: Number of records to return. Default value is None.\n        :type top: int\n        :param skip: Number of records to skip. Default value is None.\n        :type skip: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either AuthorizationServerContract or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.apimanagement.models.AuthorizationServerContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.AuthorizationServerCollection] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_by_service_request(resource_group_name=resource_group_name, service_name=service_name, subscription_id=self._config.subscription_id, filter=filter, top=top, skip=skip, api_version=api_version, template_url=self.list_by_service.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('AuthorizationServerCollection', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
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
    list_by_service.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers'}

    @distributed_trace
    def get_entity_tag(self, resource_group_name: str, service_name: str, authsid: str, **kwargs: Any) -> bool:
        if False:
            print('Hello World!')
        'Gets the entity state (Etag) version of the authorizationServer specified by its identifier.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: bool or the result of cls(response)\n        :rtype: bool\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_get_entity_tag_request(resource_group_name=resource_group_name, service_name=service_name, authsid=authsid, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_entity_tag.metadata['url'], headers=_headers, params=_params)
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
    get_entity_tag.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}'}

    @distributed_trace
    def get(self, resource_group_name: str, service_name: str, authsid: str, **kwargs: Any) -> _models.AuthorizationServerContract:
        if False:
            while True:
                i = 10
        'Gets the details of the authorization server specified by its identifier.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.AuthorizationServerContract] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, service_name=service_name, authsid=authsid, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
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
        deserialized = self._deserialize('AuthorizationServerContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}'}

    @overload
    def create_or_update(self, resource_group_name: str, service_name: str, authsid: str, parameters: _models.AuthorizationServerContract, if_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.AuthorizationServerContract:
        if False:
            i = 10
            return i + 15
        'Creates new authorization server or updates an existing authorization server.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :param parameters: Create or update parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create_or_update(self, resource_group_name: str, service_name: str, authsid: str, parameters: IO, if_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.AuthorizationServerContract:
        if False:
            for i in range(10):
                print('nop')
        'Creates new authorization server or updates an existing authorization server.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :param parameters: Create or update parameters. Required.\n        :type parameters: IO\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create_or_update(self, resource_group_name: str, service_name: str, authsid: str, parameters: Union[_models.AuthorizationServerContract, IO], if_match: Optional[str]=None, **kwargs: Any) -> _models.AuthorizationServerContract:
        if False:
            for i in range(10):
                print('nop')
        "Creates new authorization server or updates an existing authorization server.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :param parameters: Create or update parameters. Is either a AuthorizationServerContract type or\n         a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.AuthorizationServerContract or IO\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.AuthorizationServerContract] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'AuthorizationServerContract')
        request = build_create_or_update_request(resource_group_name=resource_group_name, service_name=service_name, authsid=authsid, subscription_id=self._config.subscription_id, if_match=if_match, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create_or_update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 200:
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('AuthorizationServerContract', pipeline_response)
        if response.status_code == 201:
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('AuthorizationServerContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}'}

    @overload
    def update(self, resource_group_name: str, service_name: str, authsid: str, if_match: str, parameters: _models.AuthorizationServerUpdateContract, *, content_type: str='application/json', **kwargs: Any) -> _models.AuthorizationServerContract:
        if False:
            i = 10
            return i + 15
        'Updates the details of the authorization server specified by its identifier.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :param parameters: OAuth2 Server settings Update parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.AuthorizationServerUpdateContract\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, service_name: str, authsid: str, if_match: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.AuthorizationServerContract:
        if False:
            i = 10
            return i + 15
        'Updates the details of the authorization server specified by its identifier.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :param parameters: OAuth2 Server settings Update parameters. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, service_name: str, authsid: str, if_match: str, parameters: Union[_models.AuthorizationServerUpdateContract, IO], **kwargs: Any) -> _models.AuthorizationServerContract:
        if False:
            i = 10
            return i + 15
        "Updates the details of the authorization server specified by its identifier.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :param parameters: OAuth2 Server settings Update parameters. Is either a\n         AuthorizationServerUpdateContract type or a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.AuthorizationServerUpdateContract or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.AuthorizationServerContract] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'AuthorizationServerUpdateContract')
        request = build_update_request(resource_group_name=resource_group_name, service_name=service_name, authsid=authsid, subscription_id=self._config.subscription_id, if_match=if_match, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
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
        deserialized = self._deserialize('AuthorizationServerContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}'}

    @distributed_trace
    def delete(self, resource_group_name: str, service_name: str, authsid: str, if_match: str, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Deletes specific authorization server instance.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, service_name=service_name, authsid=authsid, subscription_id=self._config.subscription_id, if_match=if_match, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
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
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}'}

    @distributed_trace
    def list_secrets(self, resource_group_name: str, service_name: str, authsid: str, **kwargs: Any) -> _models.AuthorizationServerSecretsContract:
        if False:
            return 10
        'Gets the client secret details of the authorization server.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authsid: Identifier of the authorization server. Required.\n        :type authsid: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationServerSecretsContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationServerSecretsContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.AuthorizationServerSecretsContract] = kwargs.pop('cls', None)
        request = build_list_secrets_request(resource_group_name=resource_group_name, service_name=service_name, authsid=authsid, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_secrets.metadata['url'], headers=_headers, params=_params)
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
        deserialized = self._deserialize('AuthorizationServerSecretsContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    list_secrets.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationServers/{authsid}/listSecrets'}