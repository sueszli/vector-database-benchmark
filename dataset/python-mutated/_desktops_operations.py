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
from .._serialization import Serializer
from .._vendor import _convert_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(resource_group_name: str, application_group_name: str, desktop_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/applicationGroups/{applicationGroupName}/desktops/{desktopName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'applicationGroupName': _SERIALIZER.url('application_group_name', application_group_name, 'str', max_length=64, min_length=3), 'desktopName': _SERIALIZER.url('desktop_name', desktop_name, 'str', max_length=24, min_length=3)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, application_group_name: str, desktop_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/applicationGroups/{applicationGroupName}/desktops/{desktopName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'applicationGroupName': _SERIALIZER.url('application_group_name', application_group_name, 'str', max_length=64, min_length=3), 'desktopName': _SERIALIZER.url('desktop_name', desktop_name, 'str', max_length=24, min_length=3)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_request(resource_group_name: str, application_group_name: str, subscription_id: str, *, page_size: Optional[int]=None, is_descending: Optional[bool]=None, initial_skip: Optional[int]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/applicationGroups/{applicationGroupName}/desktops')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'applicationGroupName': _SERIALIZER.url('application_group_name', application_group_name, 'str', max_length=64, min_length=3)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if page_size is not None:
        _params['pageSize'] = _SERIALIZER.query('page_size', page_size, 'int')
    if is_descending is not None:
        _params['isDescending'] = _SERIALIZER.query('is_descending', is_descending, 'bool')
    if initial_skip is not None:
        _params['initialSkip'] = _SERIALIZER.query('initial_skip', initial_skip, 'int')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class DesktopsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.desktopvirtualization.DesktopVirtualizationMgmtClient`'s
        :attr:`desktops` attribute.
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
    def get(self, resource_group_name: str, application_group_name: str, desktop_name: str, **kwargs: Any) -> _models.Desktop:
        if False:
            i = 10
            return i + 15
        'Get a desktop.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param application_group_name: The name of the application group. Required.\n        :type application_group_name: str\n        :param desktop_name: The name of the desktop within the specified desktop group. Required.\n        :type desktop_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Desktop or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.Desktop\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.Desktop] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, application_group_name=application_group_name, desktop_name=desktop_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Desktop', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/applicationGroups/{applicationGroupName}/desktops/{desktopName}'}

    @overload
    def update(self, resource_group_name: str, application_group_name: str, desktop_name: str, desktop: Optional[_models.DesktopPatch]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Desktop:
        if False:
            print('Hello World!')
        'Update a desktop.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param application_group_name: The name of the application group. Required.\n        :type application_group_name: str\n        :param desktop_name: The name of the desktop within the specified desktop group. Required.\n        :type desktop_name: str\n        :param desktop: Object containing Desktop definitions. Default value is None.\n        :type desktop: ~azure.mgmt.desktopvirtualization.models.DesktopPatch\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Desktop or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.Desktop\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, application_group_name: str, desktop_name: str, desktop: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Desktop:
        if False:
            for i in range(10):
                print('nop')
        'Update a desktop.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param application_group_name: The name of the application group. Required.\n        :type application_group_name: str\n        :param desktop_name: The name of the desktop within the specified desktop group. Required.\n        :type desktop_name: str\n        :param desktop: Object containing Desktop definitions. Default value is None.\n        :type desktop: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Desktop or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.Desktop\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, application_group_name: str, desktop_name: str, desktop: Optional[Union[_models.DesktopPatch, IO]]=None, **kwargs: Any) -> _models.Desktop:
        if False:
            print('Hello World!')
        "Update a desktop.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param application_group_name: The name of the application group. Required.\n        :type application_group_name: str\n        :param desktop_name: The name of the desktop within the specified desktop group. Required.\n        :type desktop_name: str\n        :param desktop: Object containing Desktop definitions. Is either a DesktopPatch type or a IO\n         type. Default value is None.\n        :type desktop: ~azure.mgmt.desktopvirtualization.models.DesktopPatch or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Desktop or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.Desktop\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Desktop] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(desktop, (IOBase, bytes)):
            _content = desktop
        elif desktop is not None:
            _json = self._serialize.body(desktop, 'DesktopPatch')
        else:
            _json = None
        request = build_update_request(resource_group_name=resource_group_name, application_group_name=application_group_name, desktop_name=desktop_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Desktop', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/applicationGroups/{applicationGroupName}/desktops/{desktopName}'}

    @distributed_trace
    def list(self, resource_group_name: str, application_group_name: str, page_size: Optional[int]=None, is_descending: Optional[bool]=None, initial_skip: Optional[int]=None, **kwargs: Any) -> Iterable['_models.Desktop']:
        if False:
            i = 10
            return i + 15
        'List desktops.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param application_group_name: The name of the application group. Required.\n        :type application_group_name: str\n        :param page_size: Number of items per page. Default value is None.\n        :type page_size: int\n        :param is_descending: Indicates whether the collection is descending. Default value is None.\n        :type is_descending: bool\n        :param initial_skip: Initial number of items to skip. Default value is None.\n        :type initial_skip: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Desktop or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.desktopvirtualization.models.Desktop]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.DesktopList] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_request(resource_group_name=resource_group_name, application_group_name=application_group_name, subscription_id=self._config.subscription_id, page_size=page_size, is_descending=is_descending, initial_skip=initial_skip, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('DesktopList', pipeline_response)
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
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/applicationGroups/{applicationGroupName}/desktops'}