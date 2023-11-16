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
from ..._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(resource_group_name: str, service_name: str, configuration_service_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', pattern='^[a-z][a-z0-9-]*[a-z0-9]$'), 'configurationServiceName': _SERIALIZER.url('configuration_service_name', configuration_service_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, service_name: str, configuration_service_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', pattern='^[a-z][a-z0-9-]*[a-z0-9]$'), 'configurationServiceName': _SERIALIZER.url('configuration_service_name', configuration_service_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, service_name: str, configuration_service_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', pattern='^[a-z][a-z0-9-]*[a-z0-9]$'), 'configurationServiceName': _SERIALIZER.url('configuration_service_name', configuration_service_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_request(resource_group_name: str, service_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', pattern='^[a-z][a-z0-9-]*[a-z0-9]$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_validate_request(resource_group_name: str, service_name: str, configuration_service_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}/validate')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', pattern='^[a-z][a-z0-9-]*[a-z0-9]$'), 'configurationServiceName': _SERIALIZER.url('configuration_service_name', configuration_service_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class ConfigurationServicesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.appplatform.v2022_12_01.AppPlatformManagementClient`'s
        :attr:`configuration_services` attribute.
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

    @distributed_trace
    def get(self, resource_group_name: str, service_name: str, configuration_service_name: str, **kwargs: Any) -> _models.ConfigurationServiceResource:
        if False:
            return 10
        'Get the Application Configuration Service and its properties.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ConfigurationServiceResource or the result of cls(response)\n        :rtype: ~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        cls: ClsType[_models.ConfigurationServiceResource] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, service_name=service_name, configuration_service_name=configuration_service_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ConfigurationServiceResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}'}

    def _create_or_update_initial(self, resource_group_name: str, service_name: str, configuration_service_name: str, configuration_service_resource: Union[_models.ConfigurationServiceResource, IO], **kwargs: Any) -> _models.ConfigurationServiceResource:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ConfigurationServiceResource] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(configuration_service_resource, (IO, bytes)):
            _content = configuration_service_resource
        else:
            _json = self._serialize.body(configuration_service_resource, 'ConfigurationServiceResource')
        request = build_create_or_update_request(resource_group_name=resource_group_name, service_name=service_name, configuration_service_name=configuration_service_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_or_update_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('ConfigurationServiceResource', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('ConfigurationServiceResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_or_update_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}'}

    @overload
    def begin_create_or_update(self, resource_group_name: str, service_name: str, configuration_service_name: str, configuration_service_resource: _models.ConfigurationServiceResource, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ConfigurationServiceResource]:
        if False:
            print('Hello World!')
        'Create the default Application Configuration Service or update the existing Application\n        Configuration Service.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :param configuration_service_resource: Parameters for the update operation. Required.\n        :type configuration_service_resource:\n         ~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceResource\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ConfigurationServiceResource or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_or_update(self, resource_group_name: str, service_name: str, configuration_service_name: str, configuration_service_resource: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ConfigurationServiceResource]:
        if False:
            while True:
                i = 10
        'Create the default Application Configuration Service or update the existing Application\n        Configuration Service.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :param configuration_service_resource: Parameters for the update operation. Required.\n        :type configuration_service_resource: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ConfigurationServiceResource or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_or_update(self, resource_group_name: str, service_name: str, configuration_service_name: str, configuration_service_resource: Union[_models.ConfigurationServiceResource, IO], **kwargs: Any) -> LROPoller[_models.ConfigurationServiceResource]:
        if False:
            for i in range(10):
                print('nop')
        "Create the default Application Configuration Service or update the existing Application\n        Configuration Service.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :param configuration_service_resource: Parameters for the update operation. Is either a\n         ConfigurationServiceResource type or a IO type. Required.\n        :type configuration_service_resource:\n         ~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceResource or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ConfigurationServiceResource or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ConfigurationServiceResource] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_or_update_initial(resource_group_name=resource_group_name, service_name=service_name, configuration_service_name=configuration_service_name, configuration_service_resource=configuration_service_resource, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('ConfigurationServiceResource', pipeline_response)
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
    begin_create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}'}

    def _delete_initial(self, resource_group_name: str, service_name: str, configuration_service_name: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, service_name=service_name, configuration_service_name=configuration_service_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _delete_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}'}

    @distributed_trace
    def begin_delete(self, resource_group_name: str, service_name: str, configuration_service_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            for i in range(10):
                print('nop')
        'Disable the default Application Configuration Service.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_initial(resource_group_name=resource_group_name, service_name=service_name, configuration_service_name=configuration_service_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
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
    begin_delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}'}

    @distributed_trace
    def list(self, resource_group_name: str, service_name: str, **kwargs: Any) -> Iterable['_models.ConfigurationServiceResource']:
        if False:
            print('Hello World!')
        'Handles requests to list all resources in a Service.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ConfigurationServiceResource or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        cls: ClsType[_models.ConfigurationServiceResourceCollection] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_request(resource_group_name=resource_group_name, service_name=service_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('ConfigurationServiceResourceCollection', pipeline_response)
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
    list.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices'}

    def _validate_initial(self, resource_group_name: str, service_name: str, configuration_service_name: str, settings: Union[_models.ConfigurationServiceSettings, IO], **kwargs: Any) -> _models.ConfigurationServiceSettingsValidateResult:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ConfigurationServiceSettingsValidateResult] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(settings, (IO, bytes)):
            _content = settings
        else:
            _json = self._serialize.body(settings, 'ConfigurationServiceSettings')
        request = build_validate_request(resource_group_name=resource_group_name, service_name=service_name, configuration_service_name=configuration_service_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._validate_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('ConfigurationServiceSettingsValidateResult', pipeline_response)
        if response.status_code == 202:
            deserialized = self._deserialize('ConfigurationServiceSettingsValidateResult', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _validate_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}/validate'}

    @overload
    def begin_validate(self, resource_group_name: str, service_name: str, configuration_service_name: str, settings: _models.ConfigurationServiceSettings, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ConfigurationServiceSettingsValidateResult]:
        if False:
            i = 10
            return i + 15
        'Check if the Application Configuration Service settings are valid.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :param settings: Application Configuration Service settings to be validated. Required.\n        :type settings: ~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceSettings\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either\n         ConfigurationServiceSettingsValidateResult or the result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceSettingsValidateResult]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_validate(self, resource_group_name: str, service_name: str, configuration_service_name: str, settings: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ConfigurationServiceSettingsValidateResult]:
        if False:
            for i in range(10):
                print('nop')
        'Check if the Application Configuration Service settings are valid.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :param settings: Application Configuration Service settings to be validated. Required.\n        :type settings: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either\n         ConfigurationServiceSettingsValidateResult or the result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceSettingsValidateResult]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_validate(self, resource_group_name: str, service_name: str, configuration_service_name: str, settings: Union[_models.ConfigurationServiceSettings, IO], **kwargs: Any) -> LROPoller[_models.ConfigurationServiceSettingsValidateResult]:
        if False:
            i = 10
            return i + 15
        "Check if the Application Configuration Service settings are valid.\n\n        :param resource_group_name: The name of the resource group that contains the resource. You can\n         obtain this value from the Azure Resource Manager API or the portal. Required.\n        :type resource_group_name: str\n        :param service_name: The name of the Service resource. Required.\n        :type service_name: str\n        :param configuration_service_name: The name of Application Configuration Service. Required.\n        :type configuration_service_name: str\n        :param settings: Application Configuration Service settings to be validated. Is either a\n         ConfigurationServiceSettings type or a IO type. Required.\n        :type settings: ~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceSettings or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either\n         ConfigurationServiceSettingsValidateResult or the result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appplatform.v2022_12_01.models.ConfigurationServiceSettingsValidateResult]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-12-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-12-01'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ConfigurationServiceSettingsValidateResult] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._validate_initial(resource_group_name=resource_group_name, service_name=service_name, configuration_service_name=configuration_service_name, settings=settings, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('ConfigurationServiceSettingsValidateResult', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
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
    begin_validate.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.AppPlatform/Spring/{serviceName}/configurationServices/{configurationServiceName}/validate'}