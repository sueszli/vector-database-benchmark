import sys
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
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_feature_value_request(location_id: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/providers/Microsoft.DataFactory/locations/{locationId}/getFeatureValue')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'locationId': _SERIALIZER.url('location_id', location_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_feature_value_by_factory_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/getFeatureValue')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_query_feature_values_by_factory_request(resource_group_name: str, factory_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2018-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/queryFeaturesValue')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'factoryName': _SERIALIZER.url('factory_name', factory_name, 'str', max_length=63, min_length=3, pattern='^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class ExposureControlOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.datafactory.DataFactoryManagementClient`'s
        :attr:`exposure_control` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @overload
    def get_feature_value(self, location_id: str, exposure_control_request: _models.ExposureControlRequest, *, content_type: str='application/json', **kwargs: Any) -> _models.ExposureControlResponse:
        if False:
            return 10
        'Get exposure control feature for specific location.\n\n        :param location_id: The location identifier. Required.\n        :type location_id: str\n        :param exposure_control_request: The exposure control request. Required.\n        :type exposure_control_request: ~azure.mgmt.datafactory.models.ExposureControlRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def get_feature_value(self, location_id: str, exposure_control_request: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.ExposureControlResponse:
        if False:
            return 10
        'Get exposure control feature for specific location.\n\n        :param location_id: The location identifier. Required.\n        :type location_id: str\n        :param exposure_control_request: The exposure control request. Required.\n        :type exposure_control_request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def get_feature_value(self, location_id: str, exposure_control_request: Union[_models.ExposureControlRequest, IO], **kwargs: Any) -> _models.ExposureControlResponse:
        if False:
            for i in range(10):
                print('nop')
        "Get exposure control feature for specific location.\n\n        :param location_id: The location identifier. Required.\n        :type location_id: str\n        :param exposure_control_request: The exposure control request. Is either a\n         ExposureControlRequest type or a IO type. Required.\n        :type exposure_control_request: ~azure.mgmt.datafactory.models.ExposureControlRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ExposureControlResponse] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(exposure_control_request, (IO, bytes)):
            _content = exposure_control_request
        else:
            _json = self._serialize.body(exposure_control_request, 'ExposureControlRequest')
        request = build_get_feature_value_request(location_id=location_id, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.get_feature_value.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ExposureControlResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_feature_value.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.DataFactory/locations/{locationId}/getFeatureValue'}

    @overload
    def get_feature_value_by_factory(self, resource_group_name: str, factory_name: str, exposure_control_request: _models.ExposureControlRequest, *, content_type: str='application/json', **kwargs: Any) -> _models.ExposureControlResponse:
        if False:
            while True:
                i = 10
        'Get exposure control feature for specific factory.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param exposure_control_request: The exposure control request. Required.\n        :type exposure_control_request: ~azure.mgmt.datafactory.models.ExposureControlRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def get_feature_value_by_factory(self, resource_group_name: str, factory_name: str, exposure_control_request: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.ExposureControlResponse:
        if False:
            while True:
                i = 10
        'Get exposure control feature for specific factory.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param exposure_control_request: The exposure control request. Required.\n        :type exposure_control_request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def get_feature_value_by_factory(self, resource_group_name: str, factory_name: str, exposure_control_request: Union[_models.ExposureControlRequest, IO], **kwargs: Any) -> _models.ExposureControlResponse:
        if False:
            return 10
        "Get exposure control feature for specific factory.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param exposure_control_request: The exposure control request. Is either a\n         ExposureControlRequest type or a IO type. Required.\n        :type exposure_control_request: ~azure.mgmt.datafactory.models.ExposureControlRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ExposureControlResponse] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(exposure_control_request, (IO, bytes)):
            _content = exposure_control_request
        else:
            _json = self._serialize.body(exposure_control_request, 'ExposureControlRequest')
        request = build_get_feature_value_by_factory_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.get_feature_value_by_factory.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ExposureControlResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_feature_value_by_factory.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/getFeatureValue'}

    @overload
    def query_feature_values_by_factory(self, resource_group_name: str, factory_name: str, exposure_control_batch_request: _models.ExposureControlBatchRequest, *, content_type: str='application/json', **kwargs: Any) -> _models.ExposureControlBatchResponse:
        if False:
            return 10
        'Get list of exposure control features for specific factory.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param exposure_control_batch_request: The exposure control request for list of features.\n         Required.\n        :type exposure_control_batch_request:\n         ~azure.mgmt.datafactory.models.ExposureControlBatchRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlBatchResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlBatchResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def query_feature_values_by_factory(self, resource_group_name: str, factory_name: str, exposure_control_batch_request: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.ExposureControlBatchResponse:
        if False:
            return 10
        'Get list of exposure control features for specific factory.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param exposure_control_batch_request: The exposure control request for list of features.\n         Required.\n        :type exposure_control_batch_request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlBatchResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlBatchResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def query_feature_values_by_factory(self, resource_group_name: str, factory_name: str, exposure_control_batch_request: Union[_models.ExposureControlBatchRequest, IO], **kwargs: Any) -> _models.ExposureControlBatchResponse:
        if False:
            while True:
                i = 10
        "Get list of exposure control features for specific factory.\n\n        :param resource_group_name: The resource group name. Required.\n        :type resource_group_name: str\n        :param factory_name: The factory name. Required.\n        :type factory_name: str\n        :param exposure_control_batch_request: The exposure control request for list of features. Is\n         either a ExposureControlBatchRequest type or a IO type. Required.\n        :type exposure_control_batch_request:\n         ~azure.mgmt.datafactory.models.ExposureControlBatchRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExposureControlBatchResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.datafactory.models.ExposureControlBatchResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2018-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ExposureControlBatchResponse] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(exposure_control_batch_request, (IO, bytes)):
            _content = exposure_control_batch_request
        else:
            _json = self._serialize.body(exposure_control_batch_request, 'ExposureControlBatchRequest')
        request = build_query_feature_values_by_factory_request(resource_group_name=resource_group_name, factory_name=factory_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.query_feature_values_by_factory.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ExposureControlBatchResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    query_feature_values_by_factory.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataFactory/factories/{factoryName}/queryFeaturesValue'}