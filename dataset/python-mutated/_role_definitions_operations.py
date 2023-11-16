from io import IOBase
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
from ..._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_delete_request(scope: str, role_definition_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-05-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'roleDefinitionId': _SERIALIZER.url('role_definition_id', role_definition_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(scope: str, role_definition_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-05-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'roleDefinitionId': _SERIALIZER.url('role_definition_id', role_definition_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(scope: str, role_definition_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-05-01-preview'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'roleDefinitionId': _SERIALIZER.url('role_definition_id', role_definition_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_request(scope: str, *, filter: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-05-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleDefinitions')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_by_id_request(role_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-05-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{roleId}')
    path_format_arguments = {'roleId': _SERIALIZER.url('role_id', role_id, 'str', skip_quote=True)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class RoleDefinitionsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.authorization.v2022_05_01_preview.AuthorizationManagementClient`'s
        :attr:`role_definitions` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')
        self._api_version = input_args.pop(0) if input_args else kwargs.pop('api_version')

    @distributed_trace
    def delete(self, scope: str, role_definition_id: str, **kwargs: Any) -> Optional[_models.RoleDefinition]:
        if False:
            return 10
        "Deletes a role definition.\n\n        :param scope: The scope of the operation or resource. Valid scopes are: subscription (format:\n         '/subscriptions/{subscriptionId}'), resource group (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}', or resource (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/{resourceProviderNamespace}/[{parentResourcePath}/]{resourceType}/{resourceName}'.\n         Required.\n        :type scope: str\n        :param role_definition_id: The ID of the role definition to delete. Required.\n        :type role_definition_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleDefinition or None or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition or None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2022-05-01-preview'))
        cls: ClsType[Optional[_models.RoleDefinition]] = kwargs.pop('cls', None)
        request = build_delete_request(scope=scope, role_definition_id=role_definition_id, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('RoleDefinition', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    delete.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId}'}

    @distributed_trace
    def get(self, scope: str, role_definition_id: str, **kwargs: Any) -> _models.RoleDefinition:
        if False:
            while True:
                i = 10
        "Get role definition by ID (GUID).\n\n        :param scope: The scope of the operation or resource. Valid scopes are: subscription (format:\n         '/subscriptions/{subscriptionId}'), resource group (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}', or resource (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/{resourceProviderNamespace}/[{parentResourcePath}/]{resourceType}/{resourceName}'.\n         Required.\n        :type scope: str\n        :param role_definition_id: The ID of the role definition. Required.\n        :type role_definition_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleDefinition or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2022-05-01-preview'))
        cls: ClsType[_models.RoleDefinition] = kwargs.pop('cls', None)
        request = build_get_request(scope=scope, role_definition_id=role_definition_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('RoleDefinition', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId}'}

    @overload
    def create_or_update(self, scope: str, role_definition_id: str, role_definition: _models.RoleDefinition, *, content_type: str='application/json', **kwargs: Any) -> _models.RoleDefinition:
        if False:
            print('Hello World!')
        'Creates or updates a role definition.\n\n        :param scope: The scope of the operation or resource. Valid scopes are: subscription (format:\n         \'/subscriptions/{subscriptionId}\'), resource group (format:\n         \'/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}\', or resource (format:\n         \'/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/{resourceProviderNamespace}/[{parentResourcePath}/]{resourceType}/{resourceName}\'.\n         Required.\n        :type scope: str\n        :param role_definition_id: The ID of the role definition. Required.\n        :type role_definition_id: str\n        :param role_definition: The values for the role definition. Required.\n        :type role_definition: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleDefinition or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create_or_update(self, scope: str, role_definition_id: str, role_definition: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.RoleDefinition:
        if False:
            return 10
        'Creates or updates a role definition.\n\n        :param scope: The scope of the operation or resource. Valid scopes are: subscription (format:\n         \'/subscriptions/{subscriptionId}\'), resource group (format:\n         \'/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}\', or resource (format:\n         \'/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/{resourceProviderNamespace}/[{parentResourcePath}/]{resourceType}/{resourceName}\'.\n         Required.\n        :type scope: str\n        :param role_definition_id: The ID of the role definition. Required.\n        :type role_definition_id: str\n        :param role_definition: The values for the role definition. Required.\n        :type role_definition: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleDefinition or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create_or_update(self, scope: str, role_definition_id: str, role_definition: Union[_models.RoleDefinition, IO], **kwargs: Any) -> _models.RoleDefinition:
        if False:
            return 10
        "Creates or updates a role definition.\n\n        :param scope: The scope of the operation or resource. Valid scopes are: subscription (format:\n         '/subscriptions/{subscriptionId}'), resource group (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}', or resource (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/{resourceProviderNamespace}/[{parentResourcePath}/]{resourceType}/{resourceName}'.\n         Required.\n        :type scope: str\n        :param role_definition_id: The ID of the role definition. Required.\n        :type role_definition_id: str\n        :param role_definition: The values for the role definition. Is either a RoleDefinition type or\n         a IO type. Required.\n        :type role_definition: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition or\n         IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleDefinition or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2022-05-01-preview'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.RoleDefinition] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(role_definition, (IOBase, bytes)):
            _content = role_definition
        else:
            _json = self._serialize.body(role_definition, 'RoleDefinition')
        request = build_create_or_update_request(scope=scope, role_definition_id=role_definition_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create_or_update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('RoleDefinition', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create_or_update.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId}'}

    @distributed_trace
    def list(self, scope: str, filter: Optional[str]=None, **kwargs: Any) -> Iterable['_models.RoleDefinition']:
        if False:
            while True:
                i = 10
        "Get all role definitions that are applicable at scope and above.\n\n        :param scope: The scope of the operation or resource. Valid scopes are: subscription (format:\n         '/subscriptions/{subscriptionId}'), resource group (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}', or resource (format:\n         '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/{resourceProviderNamespace}/[{parentResourcePath}/]{resourceType}/{resourceName}'.\n         Required.\n        :type scope: str\n        :param filter: The filter to apply on the operation. Use atScopeAndBelow filter to search below\n         the given scope as well. Default value is None.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either RoleDefinition or the result of cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2022-05-01-preview'))
        cls: ClsType[_models.RoleDefinitionListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_request(scope=scope, filter=filter, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('RoleDefinitionListResult', pipeline_response)
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
    list.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleDefinitions'}

    @distributed_trace
    def get_by_id(self, role_id: str, **kwargs: Any) -> _models.RoleDefinition:
        if False:
            print('Hello World!')
        'Gets a role definition by ID.\n\n        :param role_id: The fully qualified role definition ID. Use the format,\n         /subscriptions/{guid}/providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId} for\n         subscription level role definitions, or\n         /providers/Microsoft.Authorization/roleDefinitions/{roleDefinitionId} for tenant level role\n         definitions. Required.\n        :type role_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleDefinition or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2022_05_01_preview.models.RoleDefinition\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2022-05-01-preview'))
        cls: ClsType[_models.RoleDefinition] = kwargs.pop('cls', None)
        request = build_get_by_id_request(role_id=role_id, api_version=api_version, template_url=self.get_by_id.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('RoleDefinition', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_by_id.metadata = {'url': '/{roleId}'}