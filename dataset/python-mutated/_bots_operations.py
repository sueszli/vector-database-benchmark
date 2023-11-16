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

def build_create_request(resource_group_name: str, resource_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'resourceName': _SERIALIZER.url('resource_name', resource_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, resource_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'resourceName': _SERIALIZER.url('resource_name', resource_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, resource_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'resourceName': _SERIALIZER.url('resource_name', resource_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, resource_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'resourceName': _SERIALIZER.url('resource_name', resource_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_by_resource_group_request(resource_group_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_request(subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/providers/Microsoft.BotService/botServices')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_check_name_availability_request(**kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.BotService/checkNameAvailability')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class BotsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.botservice.AzureBotService`'s
        :attr:`bots` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @overload
    def create(self, resource_group_name: str, resource_name: str, parameters: _models.Bot, *, content_type: str='application/json', **kwargs: Any) -> _models.Bot:
        if False:
            i = 10
            return i + 15
        'Creates a Bot Service. Bot Service is a resource group wide resource type.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :param parameters: The parameters to provide for the created bot. Required.\n        :type parameters: ~azure.mgmt.botservice.models.Bot\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Bot or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.Bot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create(self, resource_group_name: str, resource_name: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.Bot:
        if False:
            i = 10
            return i + 15
        'Creates a Bot Service. Bot Service is a resource group wide resource type.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :param parameters: The parameters to provide for the created bot. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Bot or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.Bot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create(self, resource_group_name: str, resource_name: str, parameters: Union[_models.Bot, IO], **kwargs: Any) -> _models.Bot:
        if False:
            while True:
                i = 10
        "Creates a Bot Service. Bot Service is a resource group wide resource type.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :param parameters: The parameters to provide for the created bot. Is either a model type or a\n         IO type. Required.\n        :type parameters: ~azure.mgmt.botservice.models.Bot or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Bot or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.Bot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Bot] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'Bot')
        request = build_create_request(resource_group_name=resource_group_name, resource_name=resource_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('Bot', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('Bot', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}'}

    @distributed_trace
    def update(self, resource_group_name: str, resource_name: str, location: Optional[str]=None, tags: Optional[Dict[str, str]]=None, sku: Optional[_models.Sku]=None, kind: Optional[Union[str, _models.Kind]]=None, etag: Optional[str]=None, properties: Optional[_models.BotProperties]=None, **kwargs: Any) -> _models.Bot:
        if False:
            for i in range(10):
                print('nop')
        'Updates a Bot Service.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :param location: Specifies the location of the resource. Default value is None.\n        :type location: str\n        :param tags: Contains resource tags defined as key/value pairs. Default value is None.\n        :type tags: dict[str, str]\n        :param sku: Gets or sets the SKU of the resource. Default value is None.\n        :type sku: ~azure.mgmt.botservice.models.Sku\n        :param kind: Required. Gets or sets the Kind of the resource. Known values are: "sdk",\n         "designer", "bot", "function", and "azurebot". Default value is None.\n        :type kind: str or ~azure.mgmt.botservice.models.Kind\n        :param etag: Entity Tag. Default value is None.\n        :type etag: str\n        :param properties: The set of properties specific to bot resource. Default value is None.\n        :type properties: ~azure.mgmt.botservice.models.BotProperties\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Bot or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.Bot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: str = kwargs.pop('content_type', _headers.pop('Content-Type', 'application/json'))
        cls: ClsType[_models.Bot] = kwargs.pop('cls', None)
        _parameters = _models.Bot(etag=etag, kind=kind, location=location, properties=properties, sku=sku, tags=tags)
        _json = self._serialize.body(_parameters, 'Bot')
        request = build_update_request(resource_group_name=resource_group_name, resource_name=resource_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('Bot', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('Bot', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}'}

    @distributed_trace
    def delete(self, resource_group_name: str, resource_name: str, **kwargs: Any) -> None:
        if False:
            return 10
        'Deletes a Bot Service from the resource group.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, resource_name=resource_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}'}

    @distributed_trace
    def get(self, resource_group_name: str, resource_name: str, **kwargs: Any) -> _models.Bot:
        if False:
            for i in range(10):
                print('nop')
        'Returns a BotService specified by the parameters.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Bot or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.Bot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.Bot] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, resource_name=resource_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Bot', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}'}

    @distributed_trace
    def list_by_resource_group(self, resource_group_name: str, **kwargs: Any) -> Iterable['_models.Bot']:
        if False:
            while True:
                i = 10
        'Returns all the resources of a particular type belonging to a resource group.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Bot or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.botservice.models.Bot]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.BotResponseList] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_by_resource_group_request(resource_group_name=resource_group_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_by_resource_group.metadata['url'], headers=_headers, params=_params)
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
                i = 10
                return i + 15
            deserialized = self._deserialize('BotResponseList', pipeline_response)
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
                error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_resource_group.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices'}

    @distributed_trace
    def list(self, **kwargs: Any) -> Iterable['_models.Bot']:
        if False:
            i = 10
            return i + 15
        'Returns all the resources of a particular type belonging to a subscription.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Bot or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.botservice.models.Bot]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.BotResponseList] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_request(subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
                while True:
                    i = 10
            deserialized = self._deserialize('BotResponseList', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.BotService/botServices'}

    @overload
    def get_check_name_availability(self, parameters: _models.CheckNameAvailabilityRequestBody, *, content_type: str='application/json', **kwargs: Any) -> _models.CheckNameAvailabilityResponseBody:
        if False:
            return 10
        'Check whether a bot name is available.\n\n        :param parameters: The request body parameters to provide for the check name availability\n         request. Required.\n        :type parameters: ~azure.mgmt.botservice.models.CheckNameAvailabilityRequestBody\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CheckNameAvailabilityResponseBody or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.CheckNameAvailabilityResponseBody\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def get_check_name_availability(self, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.CheckNameAvailabilityResponseBody:
        if False:
            i = 10
            return i + 15
        'Check whether a bot name is available.\n\n        :param parameters: The request body parameters to provide for the check name availability\n         request. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CheckNameAvailabilityResponseBody or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.CheckNameAvailabilityResponseBody\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def get_check_name_availability(self, parameters: Union[_models.CheckNameAvailabilityRequestBody, IO], **kwargs: Any) -> _models.CheckNameAvailabilityResponseBody:
        if False:
            return 10
        "Check whether a bot name is available.\n\n        :param parameters: The request body parameters to provide for the check name availability\n         request. Is either a model type or a IO type. Required.\n        :type parameters: ~azure.mgmt.botservice.models.CheckNameAvailabilityRequestBody or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CheckNameAvailabilityResponseBody or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.CheckNameAvailabilityResponseBody\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.CheckNameAvailabilityResponseBody] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'CheckNameAvailabilityRequestBody')
        request = build_get_check_name_availability_request(api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.get_check_name_availability.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('CheckNameAvailabilityResponseBody', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_check_name_availability.metadata = {'url': '/providers/Microsoft.BotService/checkNameAvailability'}