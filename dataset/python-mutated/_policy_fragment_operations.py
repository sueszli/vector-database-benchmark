from typing import Any, Callable, Dict, IO, Optional, TypeVar, Union, cast, overload
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
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

def build_list_by_service_request(resource_group_name: str, service_name: str, subscription_id: str, *, filter: Optional[str]=None, orderby: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str')
    if orderby is not None:
        _params['$orderby'] = _SERIALIZER.query('orderby', orderby, 'str')
    if top is not None:
        _params['$top'] = _SERIALIZER.query('top', top, 'int', minimum=1)
    if skip is not None:
        _params['$skip'] = _SERIALIZER.query('skip', skip, 'int', minimum=0)
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_entity_tag_request(resource_group_name: str, service_name: str, id: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'id': _SERIALIZER.url('id', id, 'str', max_length=80, min_length=1, pattern='(^[\\w]+$)|(^[\\w][\\w\\-]+[\\w]$)'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, service_name: str, id: str, subscription_id: str, *, format: Optional[Union[str, _models.PolicyFragmentContentFormat]]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'id': _SERIALIZER.url('id', id, 'str', max_length=80, min_length=1, pattern='(^[\\w]+$)|(^[\\w][\\w\\-]+[\\w]$)'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if format is not None:
        _params['format'] = _SERIALIZER.query('format', format, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, service_name: str, id: str, subscription_id: str, *, if_match: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'id': _SERIALIZER.url('id', id, 'str', max_length=80, min_length=1, pattern='(^[\\w]+$)|(^[\\w][\\w\\-]+[\\w]$)'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, service_name: str, id: str, subscription_id: str, *, if_match: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'id': _SERIALIZER.url('id', id, 'str', max_length=80, min_length=1, pattern='(^[\\w]+$)|(^[\\w][\\w\\-]+[\\w]$)'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_references_request(resource_group_name: str, service_name: str, id: str, subscription_id: str, *, top: Optional[int]=None, skip: Optional[int]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}/listReferences')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'id': _SERIALIZER.url('id', id, 'str', max_length=80, min_length=1, pattern='(^[\\w]+$)|(^[\\w][\\w\\-]+[\\w]$)'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if top is not None:
        _params['$top'] = _SERIALIZER.query('top', top, 'int', minimum=1)
    if skip is not None:
        _params['$skip'] = _SERIALIZER.query('skip', skip, 'int', minimum=0)
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class PolicyFragmentOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.apimanagement.ApiManagementClient`'s
        :attr:`policy_fragment` attribute.
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
    def list_by_service(self, resource_group_name: str, service_name: str, filter: Optional[str]=None, orderby: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, **kwargs: Any) -> _models.PolicyFragmentCollection:
        if False:
            for i in range(10):
                print('nop')
        'Gets all policy fragments.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param filter: |     Field     |     Usage     |     Supported operators     |     Supported\n         functions     |</br>|-------------|-------------|-------------|-------------|</br>| name |\n         filter, orderBy | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith |</br>|\n         description | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith\n         |</br>| value | filter | ge, le, eq, ne, gt, lt | substringof, contains, startswith, endswith\n         |</br>. Default value is None.\n        :type filter: str\n        :param orderby: OData order by query option. Default value is None.\n        :type orderby: str\n        :param top: Number of records to return. Default value is None.\n        :type top: int\n        :param skip: Number of records to skip. Default value is None.\n        :type skip: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: PolicyFragmentCollection or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.PolicyFragmentCollection\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.PolicyFragmentCollection] = kwargs.pop('cls', None)
        request = build_list_by_service_request(resource_group_name=resource_group_name, service_name=service_name, subscription_id=self._config.subscription_id, filter=filter, orderby=orderby, top=top, skip=skip, api_version=api_version, template_url=self.list_by_service.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('PolicyFragmentCollection', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_by_service.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments'}

    @distributed_trace
    def get_entity_tag(self, resource_group_name: str, service_name: str, id: str, **kwargs: Any) -> bool:
        if False:
            while True:
                i = 10
        'Gets the entity state (Etag) version of a policy fragment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param id: A resource identifier. Required.\n        :type id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: bool or the result of cls(response)\n        :rtype: bool\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_get_entity_tag_request(resource_group_name=resource_group_name, service_name=service_name, id=id, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_entity_tag.metadata['url'], headers=_headers, params=_params)
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
    get_entity_tag.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}'}

    @distributed_trace
    def get(self, resource_group_name: str, service_name: str, id: str, format: Optional[Union[str, _models.PolicyFragmentContentFormat]]=None, **kwargs: Any) -> _models.PolicyFragmentContract:
        if False:
            for i in range(10):
                print('nop')
        'Gets a policy fragment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param id: A resource identifier. Required.\n        :type id: str\n        :param format: Policy fragment content format. Known values are: "xml" and "rawxml". Default\n         value is None.\n        :type format: str or ~azure.mgmt.apimanagement.models.PolicyFragmentContentFormat\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: PolicyFragmentContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.PolicyFragmentContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.PolicyFragmentContract] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, service_name=service_name, id=id, subscription_id=self._config.subscription_id, format=format, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
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
        deserialized = self._deserialize('PolicyFragmentContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}'}

    def _create_or_update_initial(self, resource_group_name: str, service_name: str, id: str, parameters: Union[_models.PolicyFragmentContract, IO], if_match: Optional[str]=None, **kwargs: Any) -> Optional[_models.PolicyFragmentContract]:
        if False:
            while True:
                i = 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.PolicyFragmentContract]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'PolicyFragmentContract')
        request = build_create_or_update_request(resource_group_name=resource_group_name, service_name=service_name, id=id, subscription_id=self._config.subscription_id, if_match=if_match, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_or_update_initial.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('PolicyFragmentContract', pipeline_response)
        if response.status_code == 201:
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('PolicyFragmentContract', pipeline_response)
        if response.status_code == 202:
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_or_update_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}'}

    @overload
    def begin_create_or_update(self, resource_group_name: str, service_name: str, id: str, parameters: _models.PolicyFragmentContract, if_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.PolicyFragmentContract]:
        if False:
            while True:
                i = 10
        'Creates or updates a policy fragment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param id: A resource identifier. Required.\n        :type id: str\n        :param parameters: The policy fragment contents to apply. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.PolicyFragmentContract\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either PolicyFragmentContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.PolicyFragmentContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_or_update(self, resource_group_name: str, service_name: str, id: str, parameters: IO, if_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.PolicyFragmentContract]:
        if False:
            while True:
                i = 10
        'Creates or updates a policy fragment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param id: A resource identifier. Required.\n        :type id: str\n        :param parameters: The policy fragment contents to apply. Required.\n        :type parameters: IO\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either PolicyFragmentContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.PolicyFragmentContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_or_update(self, resource_group_name: str, service_name: str, id: str, parameters: Union[_models.PolicyFragmentContract, IO], if_match: Optional[str]=None, **kwargs: Any) -> LROPoller[_models.PolicyFragmentContract]:
        if False:
            for i in range(10):
                print('nop')
        "Creates or updates a policy fragment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param id: A resource identifier. Required.\n        :type id: str\n        :param parameters: The policy fragment contents to apply. Is either a PolicyFragmentContract\n         type or a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.PolicyFragmentContract or IO\n        :param if_match: ETag of the Entity. Not required when creating an entity, but required when\n         updating an entity. Default value is None.\n        :type if_match: str\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either PolicyFragmentContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.PolicyFragmentContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.PolicyFragmentContract] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_or_update_initial(resource_group_name=resource_group_name, service_name=service_name, id=id, parameters=parameters, if_match=if_match, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            response_headers = {}
            response = pipeline_response.http_response
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('PolicyFragmentContract', pipeline_response)
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
    begin_create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}'}

    @distributed_trace
    def delete(self, resource_group_name: str, service_name: str, id: str, if_match: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Deletes a policy fragment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param id: A resource identifier. Required.\n        :type id: str\n        :param if_match: ETag of the Entity. ETag should match the current entity state from the header\n         response of the GET request or it should be * for unconditional update. Required.\n        :type if_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, service_name=service_name, id=id, subscription_id=self._config.subscription_id, if_match=if_match, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
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
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}'}

    @distributed_trace
    def list_references(self, resource_group_name: str, service_name: str, id: str, top: Optional[int]=None, skip: Optional[int]=None, **kwargs: Any) -> _models.ResourceCollection:
        if False:
            while True:
                i = 10
        'Lists policy resources that reference the policy fragment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param id: A resource identifier. Required.\n        :type id: str\n        :param top: Number of records to return. Default value is None.\n        :type top: int\n        :param skip: Number of records to skip. Default value is None.\n        :type skip: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ResourceCollection or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.ResourceCollection\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ResourceCollection] = kwargs.pop('cls', None)
        request = build_list_references_request(resource_group_name=resource_group_name, service_name=service_name, id=id, subscription_id=self._config.subscription_id, top=top, skip=skip, api_version=api_version, template_url=self.list_references.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ResourceCollection', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_references.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/policyFragments/{id}/listReferences'}