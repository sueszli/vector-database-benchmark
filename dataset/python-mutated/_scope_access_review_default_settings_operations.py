from io import IOBase
from typing import Any, Callable, Dict, IO, Optional, TypeVar, Union, overload
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from ..._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(scope: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleSettings/default')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_put_request(scope: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleSettings/default')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

class ScopeAccessReviewDefaultSettingsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.authorization.v2021_12_01_preview.AuthorizationManagementClient`'s
        :attr:`scope_access_review_default_settings` attribute.
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
        self._api_version = input_args.pop(0) if input_args else kwargs.pop('api_version')

    @distributed_trace
    def get(self, scope: str, **kwargs: Any) -> _models.AccessReviewDefaultSettings:
        if False:
            for i in range(10):
                print('nop')
        'Get access review default settings for the subscription.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AccessReviewDefaultSettings or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewDefaultSettings\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        cls: ClsType[_models.AccessReviewDefaultSettings] = kwargs.pop('cls', None)
        request = build_get_request(scope=scope, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('AccessReviewDefaultSettings', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleSettings/default'}

    @overload
    def put(self, scope: str, properties: _models.AccessReviewScheduleSettings, *, content_type: str='application/json', **kwargs: Any) -> _models.AccessReviewDefaultSettings:
        if False:
            i = 10
            return i + 15
        'Get access review default settings for the subscription.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param properties: Access review schedule settings. Required.\n        :type properties:\n         ~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewScheduleSettings\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AccessReviewDefaultSettings or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewDefaultSettings\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def put(self, scope: str, properties: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.AccessReviewDefaultSettings:
        if False:
            return 10
        'Get access review default settings for the subscription.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param properties: Access review schedule settings. Required.\n        :type properties: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AccessReviewDefaultSettings or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewDefaultSettings\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def put(self, scope: str, properties: Union[_models.AccessReviewScheduleSettings, IO], **kwargs: Any) -> _models.AccessReviewDefaultSettings:
        if False:
            while True:
                i = 10
        "Get access review default settings for the subscription.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param properties: Access review schedule settings. Is either a AccessReviewScheduleSettings\n         type or a IO type. Required.\n        :type properties:\n         ~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewScheduleSettings or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AccessReviewDefaultSettings or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewDefaultSettings\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.AccessReviewDefaultSettings] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(properties, (IOBase, bytes)):
            _content = properties
        else:
            _json = self._serialize.body(properties, 'AccessReviewScheduleSettings')
        request = build_put_request(scope=scope, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.put.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('AccessReviewDefaultSettings', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    put.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleSettings/default'}