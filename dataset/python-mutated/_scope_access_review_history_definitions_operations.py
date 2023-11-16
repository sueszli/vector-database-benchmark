from typing import Any, Callable, Dict, Iterable, Optional, TypeVar
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
from ..._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_request(scope: str, *, filter: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewHistoryDefinitions')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str', skip_quote=True)
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_by_id_request(scope: str, history_definition_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/accessReviewHistoryDefinitions/{historyDefinitionId}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str'), 'historyDefinitionId': _SERIALIZER.url('history_definition_id', history_definition_id, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class ScopeAccessReviewHistoryDefinitionsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.authorization.v2021_12_01_preview.AuthorizationManagementClient`'s
        :attr:`scope_access_review_history_definitions` attribute.
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
        self._api_version = input_args.pop(0) if input_args else kwargs.pop('api_version')

    @distributed_trace
    def list(self, scope: str, filter: Optional[str]=None, **kwargs: Any) -> Iterable['_models.AccessReviewHistoryDefinition']:
        if False:
            for i in range(10):
                print('nop')
        'Lists the accessReviewHistoryDefinitions available from this provider, definition instances are\n        only available for 30 days after creation.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param filter: The filter to apply on the operation. Only standard filters on definition name\n         and created date are supported. Default value is None.\n        :type filter: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either AccessReviewHistoryDefinition or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewHistoryDefinition]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        cls: ClsType[_models.AccessReviewHistoryDefinitionListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_request(scope=scope, filter=filter, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('AccessReviewHistoryDefinitionListResult', pipeline_response)
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
                error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewHistoryDefinitions'}

    @distributed_trace
    def get_by_id(self, scope: str, history_definition_id: str, **kwargs: Any) -> _models.AccessReviewHistoryDefinition:
        if False:
            for i in range(10):
                print('nop')
        'Get access review history definition by definition Id.\n\n        :param scope: The scope of the resource. Required.\n        :type scope: str\n        :param history_definition_id: The id of the access review history definition. Required.\n        :type history_definition_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AccessReviewHistoryDefinition or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2021_12_01_preview.models.AccessReviewHistoryDefinition\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2021-12-01-preview'))
        cls: ClsType[_models.AccessReviewHistoryDefinition] = kwargs.pop('cls', None)
        request = build_get_by_id_request(scope=scope, history_definition_id=history_definition_id, api_version=api_version, template_url=self.get_by_id.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorDefinition, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('AccessReviewHistoryDefinition', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_by_id.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/accessReviewHistoryDefinitions/{historyDefinitionId}'}