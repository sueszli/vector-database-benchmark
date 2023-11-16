from typing import Any, Callable, Dict, IO, Optional, TypeVar, Union, overload
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
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

def build_list_by_service_request(resource_group_name: str, service_name: str, quota_counter_key: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/quotas/{quotaCounterKey}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'quotaCounterKey': _SERIALIZER.url('quota_counter_key', quota_counter_key, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, service_name: str, quota_counter_key: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/quotas/{quotaCounterKey}')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'quotaCounterKey': _SERIALIZER.url('quota_counter_key', quota_counter_key, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

class QuotaByCounterKeysOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.apimanagement.ApiManagementClient`'s
        :attr:`quota_by_counter_keys` attribute.
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
    def list_by_service(self, resource_group_name: str, service_name: str, quota_counter_key: str, **kwargs: Any) -> _models.QuotaCounterCollection:
        if False:
            for i in range(10):
                print('nop')
        'Lists a collection of current quota counter periods associated with the counter-key configured\n        in the policy on the specified service instance. The api does not support paging yet.\n\n        .. seealso::\n           -\n        https://docs.microsoft.com/en-us/azure/api-management/api-management-howto-product-with-rules#a-namepolicies-ato-configure-call-rate-limit-and-quota-policies\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param quota_counter_key: Quota counter key identifier.This is the result of expression defined\n         in counter-key attribute of the quota-by-key policy.For Example, if you specify\n         counter-key="boo" in the policy, then it’s accessible by "boo" counter key. But if it’s defined\n         as counter-key="@("b"+"a")" then it will be accessible by "ba" key. Required.\n        :type quota_counter_key: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: QuotaCounterCollection or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.QuotaCounterCollection\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.QuotaCounterCollection] = kwargs.pop('cls', None)
        request = build_list_by_service_request(resource_group_name=resource_group_name, service_name=service_name, quota_counter_key=quota_counter_key, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_by_service.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('QuotaCounterCollection', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_by_service.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/quotas/{quotaCounterKey}'}

    @overload
    def update(self, resource_group_name: str, service_name: str, quota_counter_key: str, parameters: _models.QuotaCounterValueUpdateContract, *, content_type: str='application/json', **kwargs: Any) -> _models.QuotaCounterCollection:
        if False:
            return 10
        'Updates all the quota counter values specified with the existing quota counter key to a value\n        in the specified service instance. This should be used for reset of the quota counter values.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param quota_counter_key: Quota counter key identifier.This is the result of expression defined\n         in counter-key attribute of the quota-by-key policy.For Example, if you specify\n         counter-key="boo" in the policy, then it’s accessible by "boo" counter key. But if it’s defined\n         as counter-key="@("b"+"a")" then it will be accessible by "ba" key. Required.\n        :type quota_counter_key: str\n        :param parameters: The value of the quota counter to be applied to all quota counter periods.\n         Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.QuotaCounterValueUpdateContract\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: QuotaCounterCollection or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.QuotaCounterCollection\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, service_name: str, quota_counter_key: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.QuotaCounterCollection:
        if False:
            print('Hello World!')
        'Updates all the quota counter values specified with the existing quota counter key to a value\n        in the specified service instance. This should be used for reset of the quota counter values.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param quota_counter_key: Quota counter key identifier.This is the result of expression defined\n         in counter-key attribute of the quota-by-key policy.For Example, if you specify\n         counter-key="boo" in the policy, then it’s accessible by "boo" counter key. But if it’s defined\n         as counter-key="@("b"+"a")" then it will be accessible by "ba" key. Required.\n        :type quota_counter_key: str\n        :param parameters: The value of the quota counter to be applied to all quota counter periods.\n         Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: QuotaCounterCollection or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.QuotaCounterCollection\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, service_name: str, quota_counter_key: str, parameters: Union[_models.QuotaCounterValueUpdateContract, IO], **kwargs: Any) -> _models.QuotaCounterCollection:
        if False:
            while True:
                i = 10
        'Updates all the quota counter values specified with the existing quota counter key to a value\n        in the specified service instance. This should be used for reset of the quota counter values.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param quota_counter_key: Quota counter key identifier.This is the result of expression defined\n         in counter-key attribute of the quota-by-key policy.For Example, if you specify\n         counter-key="boo" in the policy, then it’s accessible by "boo" counter key. But if it’s defined\n         as counter-key="@("b"+"a")" then it will be accessible by "ba" key. Required.\n        :type quota_counter_key: str\n        :param parameters: The value of the quota counter to be applied to all quota counter periods.\n         Is either a QuotaCounterValueUpdateContract type or a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.QuotaCounterValueUpdateContract or IO\n        :keyword content_type: Body Parameter content-type. Known values are: \'application/json\'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: QuotaCounterCollection or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.QuotaCounterCollection\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.QuotaCounterCollection] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'QuotaCounterValueUpdateContract')
        request = build_update_request(resource_group_name=resource_group_name, service_name=service_name, quota_counter_key=quota_counter_key, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('QuotaCounterCollection', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/quotas/{quotaCounterKey}'}