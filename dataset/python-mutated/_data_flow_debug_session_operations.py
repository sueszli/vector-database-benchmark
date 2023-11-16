import sys
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
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_create_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/createDataFlowDebugSession')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_query_by_factory_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/queryDataFlowDebugSessions')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_add_data_flow_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/addDataFlowToDebugSession')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/deleteDataFlowDebugSession')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_execute_command_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/executeDataFlowDebugCommand')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class DataFlowDebugSessionOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.datafactory.DataFactoryManagementClient`'s
        :attr:`data_flow_debug_session` attribute.
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

    def _create_initial(self, resource_group_name: str, factory_name: str, request: Union[_models.CreateDataFlowDebugSessionRequest, IO], **kwargs: Any) -> Optional[_models.CreateDataFlowDebugSessionResponse]:
        if False:
            print('Hello World!')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.CreateDataFlowDebugSessionResponse]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(request, (IO, bytes)):
            _content = request
        else:
            _json = self._serialize.body(request, 'CreateDataFlowDebugSessionRequest')
        request = build_create_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('CreateDataFlowDebugSessionResponse', pipeline_response)
        if response.status_code == 202:
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/createDataFlowDebugSession'}

    @overload
    def begin_create(self, resource_group_name: str, factory_name: str, request: _models.CreateDataFlowDebugSessionRequest, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.CreateDataFlowDebugSessionResponse]:
        if False:
            return 10
        'Creates a data flow debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition. Required.\n        :type request: ~azure.mgmt.datafactory.models.CreateDataFlowDebugSessionRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either CreateDataFlowDebugSessionResponse or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.datafactory.models.CreateDataFlowDebugSessionResponse]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create(self, resource_group_name: str, factory_name: str, request: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.CreateDataFlowDebugSessionResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Creates a data flow debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition. Required.\n        :type request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either CreateDataFlowDebugSessionResponse or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.datafactory.models.CreateDataFlowDebugSessionResponse]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create(self, resource_group_name: str, factory_name: str, request: Union[_models.CreateDataFlowDebugSessionRequest, IO], **kwargs: Any) -> LROPoller[_models.CreateDataFlowDebugSessionResponse]:
        if False:
            print('Hello World!')
        "Creates a data flow debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition. Is either a\n         CreateDataFlowDebugSessionRequest type or a IO type. Required.\n        :type request: ~azure.mgmt.datafactory.models.CreateDataFlowDebugSessionRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either CreateDataFlowDebugSessionResponse or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.datafactory.models.CreateDataFlowDebugSessionResponse]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.CreateDataFlowDebugSessionResponse] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_initial(resource_group_name=resource_group_name, factory_name=factory_name, request=request, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('CreateDataFlowDebugSessionResponse', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/createDataFlowDebugSession'}

    @distributed_trace
    def query_by_factory(self, resource_group_name: str, factory_name: str, **kwargs: Any) -> Iterable['_models.DataFlowDebugSessionInfo']:
        if False:
            print('Hello World!')
        'Query all active data flow debug sessions.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either DataFlowDebugSessionInfo or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.datafactory.models.DataFlowDebugSessionInfo]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.QueryDataFlowDebugSessionsResponse] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_query_by_factory_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.query_by_factory.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('QueryDataFlowDebugSessionsResponse', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    query_by_factory.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/queryDataFlowDebugSessions'}

    @overload
    def add_data_flow(self, resource_group_name: str, factory_name: str, request: _models.DataFlowDebugPackage, *, content_type: str='application/json', **kwargs: Any) -> _models.AddDataFlowToDebugSessionResponse:
        if False:
            while True:
                i = 10
        'Add a data flow into debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition with debug content. Required.\n        :type request: ~azure.mgmt.datafactory.models.DataFlowDebugPackage\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AddDataFlowToDebugSessionResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.AddDataFlowToDebugSessionResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def add_data_flow(self, resource_group_name: str, factory_name: str, request: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.AddDataFlowToDebugSessionResponse:
        if False:
            while True:
                i = 10
        'Add a data flow into debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition with debug content. Required.\n        :type request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AddDataFlowToDebugSessionResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.AddDataFlowToDebugSessionResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def add_data_flow(self, resource_group_name: str, factory_name: str, request: Union[_models.DataFlowDebugPackage, IO], **kwargs: Any) -> _models.AddDataFlowToDebugSessionResponse:
        if False:
            return 10
        "Add a data flow into debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition with debug content. Is either a\n         DataFlowDebugPackage type or a IO type. Required.\n        :type request: ~azure.mgmt.datafactory.models.DataFlowDebugPackage or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AddDataFlowToDebugSessionResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.AddDataFlowToDebugSessionResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.AddDataFlowToDebugSessionResponse] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(request, (IO, bytes)):
            _content = request
        else:
            _json = self._serialize.body(request, 'DataFlowDebugPackage')
        request = build_add_data_flow_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.add_data_flow.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('AddDataFlowToDebugSessionResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    add_data_flow.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/addDataFlowToDebugSession'}

    @overload
    def delete(self, resource_group_name: str, factory_name: str, request: _models.DeleteDataFlowDebugSessionRequest, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Deletes a data flow debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition for deletion. Required.\n        :type request: ~azure.mgmt.datafactory.models.DeleteDataFlowDebugSessionRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def delete(self, resource_group_name: str, factory_name: str, request: IO, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Deletes a data flow debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition for deletion. Required.\n        :type request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def delete(self, resource_group_name: str, factory_name: str, request: Union[_models.DeleteDataFlowDebugSessionRequest, IO], **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        "Deletes a data flow debug session.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug session definition for deletion. Is either a\n         DeleteDataFlowDebugSessionRequest type or a IO type. Required.\n        :type request: ~azure.mgmt.datafactory.models.DeleteDataFlowDebugSessionRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[None] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(request, (IO, bytes)):
            _content = request
        else:
            _json = self._serialize.body(request, 'DeleteDataFlowDebugSessionRequest')
        request = build_delete_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/deleteDataFlowDebugSession'}

    def _execute_command_initial(self, resource_group_name: str, factory_name: str, request: Union[_models.DataFlowDebugCommandRequest, IO], **kwargs: Any) -> Optional[_models.DataFlowDebugCommandResponse]:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.DataFlowDebugCommandResponse]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(request, (IO, bytes)):
            _content = request
        else:
            _json = self._serialize.body(request, 'DataFlowDebugCommandRequest')
        request = build_execute_command_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._execute_command_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('DataFlowDebugCommandResponse', pipeline_response)
        if response.status_code == 202:
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _execute_command_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/executeDataFlowDebugCommand'}

    @overload
    def begin_execute_command(self, resource_group_name: str, factory_name: str, request: _models.DataFlowDebugCommandRequest, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.DataFlowDebugCommandResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Execute a data flow debug command.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug command definition. Required.\n        :type request: ~azure.mgmt.datafactory.models.DataFlowDebugCommandRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either DataFlowDebugCommandResponse or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.datafactory.models.DataFlowDebugCommandResponse]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_execute_command(self, resource_group_name: str, factory_name: str, request: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.DataFlowDebugCommandResponse]:
        if False:
            while True:
                i = 10
        'Execute a data flow debug command.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug command definition. Required.\n        :type request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either DataFlowDebugCommandResponse or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.datafactory.models.DataFlowDebugCommandResponse]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_execute_command(self, resource_group_name: str, factory_name: str, request: Union[_models.DataFlowDebugCommandRequest, IO], **kwargs: Any) -> LROPoller[_models.DataFlowDebugCommandResponse]:
        if False:
            print('Hello World!')
        "Execute a data flow debug command.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param request: Data flow debug command definition. Is either a DataFlowDebugCommandRequest\n         type or a IO type. Required.\n        :type request: ~azure.mgmt.datafactory.models.DataFlowDebugCommandRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either DataFlowDebugCommandResponse or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.datafactory.models.DataFlowDebugCommandResponse]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.DataFlowDebugCommandResponse] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._execute_command_initial(resource_group_name=resource_group_name, factory_name=factory_name, request=request, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('DataFlowDebugCommandResponse', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_execute_command.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/executeDataFlowDebugCommand'}