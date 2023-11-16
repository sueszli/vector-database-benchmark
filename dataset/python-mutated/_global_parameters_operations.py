import sys
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
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_by_factory_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, factory_name: str, global_parameter_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters/{globalParameterName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$'), 'globalParameterName': _SERIALIZER.url('global_parameter_name', global_parameter_name, 'str', max_length=260, min_length=1, pattern='^[A-Za-z0-9_][^<>*#.%&:\\\\+?/]*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, factory_name: str, global_parameter_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters/{globalParameterName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$'), 'globalParameterName': _SERIALIZER.url('global_parameter_name', global_parameter_name, 'str', max_length=260, min_length=1, pattern='^[A-Za-z0-9_][^<>*#.%&:\\\\+?/]*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, factory_name: str, global_parameter_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters/{globalParameterName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$'), 'globalParameterName': _SERIALIZER.url('global_parameter_name', global_parameter_name, 'str', max_length=260, min_length=1, pattern='^[A-Za-z0-9_][^<>*#.%&:\\\\+?/]*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

class GlobalParametersOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.datafactory.DataFactoryManagementClient`'s
        :attr:`global_parameters` attribute.
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

    @distributed_trace
    def list_by_factory(self, resource_group_name: str, factory_name: str, **kwargs: Any) -> Iterable['_models.GlobalParameterResource']:
        if False:
            for i in range(10):
                print('nop')
        'Lists Global parameters.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either GlobalParameterResource or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.datafactory.models.GlobalParameterResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.GlobalParameterListResponse] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_by_factory_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_by_factory.metadata['url'], headers=_headers, params=_params)
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
                print('Hello World!')
            deserialized = self._deserialize('GlobalParameterListResponse', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_factory.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters'}

    @distributed_trace
    def get(self, resource_group_name: str, factory_name: str, global_parameter_name: str, **kwargs: Any) -> _models.GlobalParameterResource:
        if False:
            for i in range(10):
                print('nop')
        'Gets a Global parameter.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param global_parameter_name: The global parameter name. Required.\n        :type global_parameter_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: GlobalParameterResource or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.GlobalParameterResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.GlobalParameterResource] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, factory_name=factory_name, global_parameter_name=global_parameter_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('GlobalParameterResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters/{globalParameterName}'}

    @overload
    def create_or_update(self, resource_group_name: str, factory_name: str, global_parameter_name: str, default: _models.GlobalParameterResource, *, content_type: str='application/json', **kwargs: Any) -> _models.GlobalParameterResource:
        if False:
            for i in range(10):
                print('nop')
        'Creates or updates a Global parameter.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param global_parameter_name: The global parameter name. Required.\n        :type global_parameter_name: str\n        :param default: Global parameter resource definition. Required.\n        :type default: ~azure.mgmt.datafactory.models.GlobalParameterResource\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: GlobalParameterResource or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.GlobalParameterResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create_or_update(self, resource_group_name: str, factory_name: str, global_parameter_name: str, default: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.GlobalParameterResource:
        if False:
            return 10
        'Creates or updates a Global parameter.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param global_parameter_name: The global parameter name. Required.\n        :type global_parameter_name: str\n        :param default: Global parameter resource definition. Required.\n        :type default: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: GlobalParameterResource or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.GlobalParameterResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create_or_update(self, resource_group_name: str, factory_name: str, global_parameter_name: str, default: Union[_models.GlobalParameterResource, IO], **kwargs: Any) -> _models.GlobalParameterResource:
        if False:
            for i in range(10):
                print('nop')
        "Creates or updates a Global parameter.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param global_parameter_name: The global parameter name. Required.\n        :type global_parameter_name: str\n        :param default: Global parameter resource definition. Is either a GlobalParameterResource type\n         or a IO type. Required.\n        :type default: ~azure.mgmt.datafactory.models.GlobalParameterResource or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: GlobalParameterResource or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.GlobalParameterResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.GlobalParameterResource] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(default, (IO, bytes)):
            _content = default
        else:
            _json = self._serialize.body(default, 'GlobalParameterResource')
        request = build_create_or_update_request(resource_group_name=resource_group_name, factory_name=factory_name, global_parameter_name=global_parameter_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create_or_update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('GlobalParameterResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters/{globalParameterName}'}

    @distributed_trace
    def delete(self, resource_group_name: str, factory_name: str, global_parameter_name: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Deletes a Global parameter.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param global_parameter_name: The global parameter name. Required.\n        :type global_parameter_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, factory_name=factory_name, global_parameter_name=global_parameter_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/globalParameters/{globalParameterName}'}