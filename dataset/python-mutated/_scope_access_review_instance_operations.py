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

def build_stop_request(scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/stop')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str'), 'scheduleDefinitionId': _SERIALIZER.url('schedule_definition_id', schedule_definition_id, 'str'), 'id': _SERIALIZER.url('id', id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_record_all_decisions_request(scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/recordAllDecisions')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str'), 'scheduleDefinitionId': _SERIALIZER.url('schedule_definition_id', schedule_definition_id, 'str'), 'id': _SERIALIZER.url('id', id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_reset_decisions_request(scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/resetDecisions')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str'), 'scheduleDefinitionId': _SERIALIZER.url('schedule_definition_id', schedule_definition_id, 'str'), 'id': _SERIALIZER.url('id', id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_apply_decisions_request(scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/applyDecisions')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str'), 'scheduleDefinitionId': _SERIALIZER.url('schedule_definition_id', schedule_definition_id, 'str'), 'id': _SERIALIZER.url('id', id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_send_reminders_request(scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/sendReminders')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str'), 'scheduleDefinitionId': _SERIALIZER.url('schedule_definition_id', schedule_definition_id, 'str'), 'id': _SERIALIZER.url('id', id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class ScopeAccessReviewInstanceOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.authorization.v2021_12_01_preview.AuthorizationManagementClient`'s
        :attr:`scope_access_review_instance` attribute.
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
        self._api_version = input_args.pop(0) if input_args else kwargs.pop('api_version')

    @distributed_trace
    def stop(self, scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> None:
        if False:
            return 10
        'An action to stop an access review instance.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param schedule_definition_id: The id of the access review schedule definition. Required.\n        :type schedule_definition_id: str\n        :param id: The id of the access review instance. Required.\n        :type id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_stop_request(scope=scope, schedule_definition_id=schedule_definition_id, id=id, api_version=api_version, template_url=self.stop.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    stop.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/stop'}

    @overload
    def record_all_decisions(self, scope: str, schedule_definition_id: str, id: str, properties: _models.RecordAllDecisionsProperties, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'An action to approve/deny all decisions for a review with certain filters.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param schedule_definition_id: The id of the access review schedule definition. Required.\n        :type schedule_definition_id: str\n        :param id: The id of the access review instance. Required.\n        :type id: str\n        :param properties: Record all decisions payload. Required.\n        :type properties:\n         ~azure.mgmt.authorization.v2021_12_01_preview.models.RecordAllDecisionsProperties\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def record_all_decisions(self, scope: str, schedule_definition_id: str, id: str, properties: IO, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'An action to approve/deny all decisions for a review with certain filters.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param schedule_definition_id: The id of the access review schedule definition. Required.\n        :type schedule_definition_id: str\n        :param id: The id of the access review instance. Required.\n        :type id: str\n        :param properties: Record all decisions payload. Required.\n        :type properties: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def record_all_decisions(self, scope: str, schedule_definition_id: str, id: str, properties: Union[_models.RecordAllDecisionsProperties, IO], **kwargs: Any) -> None:
        if False:
            return 10
        "An action to approve/deny all decisions for a review with certain filters.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param schedule_definition_id: The id of the access review schedule definition. Required.\n        :type schedule_definition_id: str\n        :param id: The id of the access review instance. Required.\n        :type id: str\n        :param properties: Record all decisions payload. Is either a RecordAllDecisionsProperties type\n         or a IO type. Required.\n        :type properties:\n         ~azure.mgmt.authorization.v2021_12_01_preview.models.RecordAllDecisionsProperties or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[None] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(properties, (IOBase, bytes)):
            _content = properties
        else:
            _json = self._serialize.body(properties, 'RecordAllDecisionsProperties')
        request = build_record_all_decisions_request(scope=scope, schedule_definition_id=schedule_definition_id, id=id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.record_all_decisions.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    record_all_decisions.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/recordAllDecisions'}

    @distributed_trace
    def reset_decisions(self, scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'An action to reset all decisions for an access review instance.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param schedule_definition_id: The id of the access review schedule definition. Required.\n        :type schedule_definition_id: str\n        :param id: The id of the access review instance. Required.\n        :type id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_reset_decisions_request(scope=scope, schedule_definition_id=schedule_definition_id, id=id, api_version=api_version, template_url=self.reset_decisions.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    reset_decisions.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/resetDecisions'}

    @distributed_trace
    def apply_decisions(self, scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'An action to apply all decisions for an access review instance.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param schedule_definition_id: The id of the access review schedule definition. Required.\n        :type schedule_definition_id: str\n        :param id: The id of the access review instance. Required.\n        :type id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_apply_decisions_request(scope=scope, schedule_definition_id=schedule_definition_id, id=id, api_version=api_version, template_url=self.apply_decisions.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    apply_decisions.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/applyDecisions'}

    @distributed_trace
    def send_reminders(self, scope: str, schedule_definition_id: str, id: str, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'An action to send reminders for an access review instance.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param schedule_definition_id: The id of the access review schedule definition. Required.\n        :type schedule_definition_id: str\n        :param id: The id of the access review instance. Required.\n        :type id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_send_reminders_request(scope=scope, schedule_definition_id=schedule_definition_id, id=id, api_version=api_version, template_url=self.send_reminders.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    send_reminders.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewScheduleDefinitions/{scheduleDefinitionId}/instances/{id}/sendReminders'}