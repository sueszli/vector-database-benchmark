import sys
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
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(report_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2022-11-16-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}')
    path_format_arguments = {'reportName': _SERIALIZER.url('report_name', report_name, 'str', pattern='^[-a-zA-Z0-9_]+$')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(report_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2022-11-16-preview'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}')
    path_format_arguments = {'reportName': _SERIALIZER.url('report_name', report_name, 'str', pattern='^[-a-zA-Z0-9_]+$')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(report_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2022-11-16-preview'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}')
    path_format_arguments = {'reportName': _SERIALIZER.url('report_name', report_name, 'str', pattern='^[-a-zA-Z0-9_]+$')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(report_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2022-11-16-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}')
    path_format_arguments = {'reportName': _SERIALIZER.url('report_name', report_name, 'str', pattern='^[-a-zA-Z0-9_]+$')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

class ReportOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.appcomplianceautomation.AppComplianceAutomationToolForMicrosoft365`'s
        :attr:`report` attribute.
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

    @distributed_trace
    def get(self, report_name: str, **kwargs: Any) -> _models.ReportResource:
        if False:
            for i in range(10):
                print('nop')
        'Get the AppComplianceAutomation report and its properties.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ReportResource or the result of cls(response)\n        :rtype: ~azure.mgmt.appcomplianceautomation.models.ReportResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        request = build_get_request(report_name=report_name, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ReportResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}'}

    def _create_or_update_initial(self, report_name: str, parameters: Union[_models.ReportResource, IO], **kwargs: Any) -> _models.ReportResource:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'ReportResource')
        request = build_create_or_update_request(report_name=report_name, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_or_update_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('ReportResource', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('ReportResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_or_update_initial.metadata = {'url': '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}'}

    @overload
    def begin_create_or_update(self, report_name: str, parameters: _models.ReportResource, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ReportResource]:
        if False:
            for i in range(10):
                print('nop')
        'Create a new AppComplianceAutomation report or update an exiting AppComplianceAutomation\n        report.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :param parameters: Parameters for the create or update operation. Required.\n        :type parameters: ~azure.mgmt.appcomplianceautomation.models.ReportResource\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ReportResource or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appcomplianceautomation.models.ReportResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_or_update(self, report_name: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ReportResource]:
        if False:
            for i in range(10):
                print('nop')
        'Create a new AppComplianceAutomation report or update an exiting AppComplianceAutomation\n        report.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :param parameters: Parameters for the create or update operation. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ReportResource or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appcomplianceautomation.models.ReportResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_or_update(self, report_name: str, parameters: Union[_models.ReportResource, IO], **kwargs: Any) -> LROPoller[_models.ReportResource]:
        if False:
            for i in range(10):
                print('nop')
        "Create a new AppComplianceAutomation report or update an exiting AppComplianceAutomation\n        report.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :param parameters: Parameters for the create or update operation. Is either a model type or a\n         IO type. Required.\n        :type parameters: ~azure.mgmt.appcomplianceautomation.models.ReportResource or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ReportResource or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appcomplianceautomation.models.ReportResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        polling = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_or_update_initial(report_name=report_name, parameters=parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('ReportResource', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'azure-async-operation'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_or_update.metadata = {'url': '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}'}

    def _update_initial(self, report_name: str, parameters: Union[_models.ReportResourcePatch, IO], **kwargs: Any) -> Optional[_models.ReportResource]:
        if False:
            print('Hello World!')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'ReportResourcePatch')
        request = build_update_request(report_name=report_name, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._update_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ReportResource', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('ReportResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _update_initial.metadata = {'url': '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}'}

    @overload
    def begin_update(self, report_name: str, parameters: _models.ReportResourcePatch, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ReportResource]:
        if False:
            i = 10
            return i + 15
        'Update an exiting AppComplianceAutomation report.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :param parameters: Parameters for the create or update operation. Required.\n        :type parameters: ~azure.mgmt.appcomplianceautomation.models.ReportResourcePatch\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ReportResource or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appcomplianceautomation.models.ReportResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_update(self, report_name: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ReportResource]:
        if False:
            print('Hello World!')
        'Update an exiting AppComplianceAutomation report.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :param parameters: Parameters for the create or update operation. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ReportResource or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appcomplianceautomation.models.ReportResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_update(self, report_name: str, parameters: Union[_models.ReportResourcePatch, IO], **kwargs: Any) -> LROPoller[_models.ReportResource]:
        if False:
            return 10
        "Update an exiting AppComplianceAutomation report.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :param parameters: Parameters for the create or update operation. Is either a model type or a\n         IO type. Required.\n        :type parameters: ~azure.mgmt.appcomplianceautomation.models.ReportResourcePatch or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ReportResource or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.appcomplianceautomation.models.ReportResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        polling = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._update_initial(report_name=report_name, parameters=parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('ReportResource', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'azure-async-operation'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_update.metadata = {'url': '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}'}

    def _delete_initial(self, report_name: str, **kwargs: Any) -> None:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        request = build_delete_request(report_name=report_name, api_version=api_version, template_url=self._delete_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _delete_initial.metadata = {'url': '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}'}

    @distributed_trace
    def begin_delete(self, report_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            while True:
                i = 10
        'Delete an AppComplianceAutomation report.\n\n        :param report_name: Report Name. Required.\n        :type report_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        polling = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_initial(report_name=report_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'azure-async-operation'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete.metadata = {'url': '/providers/Microsoft.AppComplianceAutomation/reports/{reportName}'}