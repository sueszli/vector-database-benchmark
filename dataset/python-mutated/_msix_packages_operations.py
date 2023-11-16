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

def build_get_request(resource_group_name: str, host_pool_name: str, msix_package_full_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'hostPoolName': _SERIALIZER.url('host_pool_name', host_pool_name, 'str', max_length=64, min_length=3), 'msixPackageFullName': _SERIALIZER.url('msix_package_full_name', msix_package_full_name, 'str', max_length=100, min_length=3)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, host_pool_name: str, msix_package_full_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'hostPoolName': _SERIALIZER.url('host_pool_name', host_pool_name, 'str', max_length=64, min_length=3), 'msixPackageFullName': _SERIALIZER.url('msix_package_full_name', msix_package_full_name, 'str', max_length=100, min_length=3)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, host_pool_name: str, msix_package_full_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'hostPoolName': _SERIALIZER.url('host_pool_name', host_pool_name, 'str', max_length=64, min_length=3), 'msixPackageFullName': _SERIALIZER.url('msix_package_full_name', msix_package_full_name, 'str', max_length=100, min_length=3)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, host_pool_name: str, msix_package_full_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'hostPoolName': _SERIALIZER.url('host_pool_name', host_pool_name, 'str', max_length=64, min_length=3), 'msixPackageFullName': _SERIALIZER.url('msix_package_full_name', msix_package_full_name, 'str', max_length=100, min_length=3)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_request(resource_group_name: str, host_pool_name: str, subscription_id: str, *, page_size: Optional[int]=None, is_descending: Optional[bool]=None, initial_skip: Optional[int]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-05'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'hostPoolName': _SERIALIZER.url('host_pool_name', host_pool_name, 'str', max_length=64, min_length=3)}
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

class MSIXPackagesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.desktopvirtualization.DesktopVirtualizationMgmtClient`'s
        :attr:`msix_packages` attribute.
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

    @distributed_trace
    def get(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, **kwargs: Any) -> _models.MSIXPackage:
        if False:
            print('Hello World!')
        'Get a msixpackage.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: MSIXPackage or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.MSIXPackage] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, host_pool_name=host_pool_name, msix_package_full_name=msix_package_full_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('MSIXPackage', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}'}

    @overload
    def create_or_update(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, msix_package: _models.MSIXPackage, *, content_type: str='application/json', **kwargs: Any) -> _models.MSIXPackage:
        if False:
            return 10
        'Create or update a MSIX package.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :param msix_package: Object containing  MSIX Package definitions. Required.\n        :type msix_package: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: MSIXPackage or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create_or_update(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, msix_package: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.MSIXPackage:
        if False:
            return 10
        'Create or update a MSIX package.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :param msix_package: Object containing  MSIX Package definitions. Required.\n        :type msix_package: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: MSIXPackage or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create_or_update(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, msix_package: Union[_models.MSIXPackage, IO], **kwargs: Any) -> _models.MSIXPackage:
        if False:
            i = 10
            return i + 15
        "Create or update a MSIX package.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :param msix_package: Object containing  MSIX Package definitions. Is either a MSIXPackage type\n         or a IO type. Required.\n        :type msix_package: ~azure.mgmt.desktopvirtualization.models.MSIXPackage or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: MSIXPackage or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.MSIXPackage] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(msix_package, (IOBase, bytes)):
            _content = msix_package
        else:
            _json = self._serialize.body(msix_package, 'MSIXPackage')
        request = build_create_or_update_request(resource_group_name=resource_group_name, host_pool_name=host_pool_name, msix_package_full_name=msix_package_full_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create_or_update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('MSIXPackage', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('MSIXPackage', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}'}

    @distributed_trace
    def delete(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Remove an MSIX Package.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, host_pool_name=host_pool_name, msix_package_full_name=msix_package_full_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}'}

    @overload
    def update(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, msix_package: Optional[_models.MSIXPackagePatch]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.MSIXPackage:
        if False:
            for i in range(10):
                print('nop')
        'Update an  MSIX Package.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :param msix_package: Object containing MSIX Package definitions. Default value is None.\n        :type msix_package: ~azure.mgmt.desktopvirtualization.models.MSIXPackagePatch\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: MSIXPackage or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, msix_package: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.MSIXPackage:
        if False:
            i = 10
            return i + 15
        'Update an  MSIX Package.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :param msix_package: Object containing MSIX Package definitions. Default value is None.\n        :type msix_package: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: MSIXPackage or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, host_pool_name: str, msix_package_full_name: str, msix_package: Optional[Union[_models.MSIXPackagePatch, IO]]=None, **kwargs: Any) -> _models.MSIXPackage:
        if False:
            return 10
        "Update an  MSIX Package.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param msix_package_full_name: The version specific package full name of the MSIX package\n         within specified hostpool. Required.\n        :type msix_package_full_name: str\n        :param msix_package: Object containing MSIX Package definitions. Is either a MSIXPackagePatch\n         type or a IO type. Default value is None.\n        :type msix_package: ~azure.mgmt.desktopvirtualization.models.MSIXPackagePatch or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: MSIXPackage or the result of cls(response)\n        :rtype: ~azure.mgmt.desktopvirtualization.models.MSIXPackage\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.MSIXPackage] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(msix_package, (IOBase, bytes)):
            _content = msix_package
        elif msix_package is not None:
            _json = self._serialize.body(msix_package, 'MSIXPackagePatch')
        else:
            _json = None
        request = build_update_request(resource_group_name=resource_group_name, host_pool_name=host_pool_name, msix_package_full_name=msix_package_full_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('MSIXPackage', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages/{msixPackageFullName}'}

    @distributed_trace
    def list(self, resource_group_name: str, host_pool_name: str, page_size: Optional[int]=None, is_descending: Optional[bool]=None, initial_skip: Optional[int]=None, **kwargs: Any) -> Iterable['_models.MSIXPackage']:
        if False:
            print('Hello World!')
        'List MSIX packages in hostpool.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param host_pool_name: The name of the host pool within the specified resource group. Required.\n        :type host_pool_name: str\n        :param page_size: Number of items per page. Default value is None.\n        :type page_size: int\n        :param is_descending: Indicates whether the collection is descending. Default value is None.\n        :type is_descending: bool\n        :param initial_skip: Initial number of items to skip. Default value is None.\n        :type initial_skip: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either MSIXPackage or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.desktopvirtualization.models.MSIXPackage]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.MSIXPackageList] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                request = build_list_request(resource_group_name=resource_group_name, host_pool_name=host_pool_name, subscription_id=self._config.subscription_id, page_size=page_size, is_descending=is_descending, initial_skip=initial_skip, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('MSIXPackageList', pipeline_response)
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
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/Microsoft.DesktopVirtualization/hostPools/{hostPoolName}/msixPackages'}