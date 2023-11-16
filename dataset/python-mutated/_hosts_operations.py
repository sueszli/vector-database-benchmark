from io import IOBase
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
from .._vendor import _convert_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_create_request(resource_group_name: str, host_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'hostName': _SERIALIZER.url('host_name', host_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, host_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'hostName': _SERIALIZER.url('host_name', host_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, host_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'hostName': _SERIALIZER.url('host_name', host_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, host_name: str, subscription_id: str, *, force: Optional[bool]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'hostName': _SERIALIZER.url('host_name', host_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if force is not None:
        _params['force'] = _SERIALIZER.query('force', force, 'bool')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_request(subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/providers/Microsoft.ConnectedVMwarevSphere/hosts')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_by_resource_group_request(resource_group_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class HostsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.connectedvmware.ConnectedVMwareMgmtClient`'s
        :attr:`hosts` attribute.
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

    def _create_initial(self, resource_group_name: str, host_name: str, body: Optional[Union[_models.Host, IO]]=None, **kwargs: Any) -> _models.Host:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Host] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(body, (IOBase, bytes)):
            _content = body
        elif body is not None:
            _json = self._serialize.body(body, 'Host')
        else:
            _json = None
        request = build_create_request(resource_group_name=resource_group_name, host_name=host_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('Host', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('Host', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}'}

    @overload
    def begin_create(self, resource_group_name: str, host_name: str, body: Optional[_models.Host]=None, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.Host]:
        if False:
            print('Hello World!')
        'Implements host PUT method.\n\n        Create Or Update host.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :param body: Request payload. Default value is None.\n        :type body: ~azure.mgmt.connectedvmware.models.Host\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Host or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.connectedvmware.models.Host]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create(self, resource_group_name: str, host_name: str, body: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.Host]:
        if False:
            i = 10
            return i + 15
        'Implements host PUT method.\n\n        Create Or Update host.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :param body: Request payload. Default value is None.\n        :type body: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Host or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.connectedvmware.models.Host]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create(self, resource_group_name: str, host_name: str, body: Optional[Union[_models.Host, IO]]=None, **kwargs: Any) -> LROPoller[_models.Host]:
        if False:
            return 10
        "Implements host PUT method.\n\n        Create Or Update host.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :param body: Request payload. Is either a Host type or a IO type. Default value is None.\n        :type body: ~azure.mgmt.connectedvmware.models.Host or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Host or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.connectedvmware.models.Host]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Host] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_initial(resource_group_name=resource_group_name, host_name=host_name, body=body, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('Host', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'azure-async-operation'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}'}

    @distributed_trace
    def get(self, resource_group_name: str, host_name: str, **kwargs: Any) -> _models.Host:
        if False:
            while True:
                i = 10
        'Gets a host.\n\n        Implements host GET method.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Host or the result of cls(response)\n        :rtype: ~azure.mgmt.connectedvmware.models.Host\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.Host] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, host_name=host_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Host', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}'}

    @overload
    def update(self, resource_group_name: str, host_name: str, body: Optional[_models.ResourcePatch]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Host:
        if False:
            return 10
        'Updates a host.\n\n        API to update certain properties of the host resource.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :param body: Resource properties to update. Default value is None.\n        :type body: ~azure.mgmt.connectedvmware.models.ResourcePatch\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Host or the result of cls(response)\n        :rtype: ~azure.mgmt.connectedvmware.models.Host\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, host_name: str, body: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Host:
        if False:
            i = 10
            return i + 15
        'Updates a host.\n\n        API to update certain properties of the host resource.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :param body: Resource properties to update. Default value is None.\n        :type body: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Host or the result of cls(response)\n        :rtype: ~azure.mgmt.connectedvmware.models.Host\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, host_name: str, body: Optional[Union[_models.ResourcePatch, IO]]=None, **kwargs: Any) -> _models.Host:
        if False:
            return 10
        "Updates a host.\n\n        API to update certain properties of the host resource.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :param body: Resource properties to update. Is either a ResourcePatch type or a IO type.\n         Default value is None.\n        :type body: ~azure.mgmt.connectedvmware.models.ResourcePatch or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Host or the result of cls(response)\n        :rtype: ~azure.mgmt.connectedvmware.models.Host\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Host] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(body, (IOBase, bytes)):
            _content = body
        elif body is not None:
            _json = self._serialize.body(body, 'ResourcePatch')
        else:
            _json = None
        request = build_update_request(resource_group_name=resource_group_name, host_name=host_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Host', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}'}

    def _delete_initial(self, resource_group_name: str, host_name: str, force: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, host_name=host_name, subscription_id=self._config.subscription_id, force=force, api_version=api_version, template_url=self._delete_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 202:
            response_headers['Location'] = self._deserialize('str', response.headers.get('Location'))
        if cls:
            return cls(pipeline_response, None, response_headers)
    _delete_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}'}

    @distributed_trace
    def begin_delete(self, resource_group_name: str, host_name: str, force: Optional[bool]=None, **kwargs: Any) -> LROPoller[None]:
        if False:
            for i in range(10):
                print('nop')
        'Deletes an host.\n\n        Implements host DELETE method.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param host_name: Name of the host. Required.\n        :type host_name: str\n        :param force: Whether force delete was specified. Default value is None.\n        :type force: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_initial(resource_group_name=resource_group_name, host_name=host_name, force=force, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts/{hostName}'}

    @distributed_trace
    def list(self, **kwargs: Any) -> Iterable['_models.Host']:
        if False:
            return 10
        'Implements GET hosts in a subscription.\n\n        List of hosts in a subscription.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Host or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.connectedvmware.models.Host]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.HostsList] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
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
                print('Hello World!')
            deserialized = self._deserialize('HostsList', pipeline_response)
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
    list.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.ConnectedVMwarevSphere/hosts'}

    @distributed_trace
    def list_by_resource_group(self, resource_group_name: str, **kwargs: Any) -> Iterable['_models.Host']:
        if False:
            return 10
        'Implements GET hosts in a resource group.\n\n        List of hosts in a resource group.\n\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Host or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.connectedvmware.models.Host]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.HostsList] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
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
                print('Hello World!')
            deserialized = self._deserialize('HostsList', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                print('Hello World!')
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
    list_by_resource_group.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ConnectedVMwarevSphere/hosts'}