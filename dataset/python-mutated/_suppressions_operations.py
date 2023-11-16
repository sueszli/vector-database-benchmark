import sys
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
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(resource_uri: str, recommendation_id: str, name: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-01-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{resourceUri}/providers/Microsoft.Advisor/recommendations/{recommendationId}/suppressions/{name}')
    path_format_arguments = {'resourceUri': _SERIALIZER.url('resource_uri', resource_uri, 'str'), 'recommendationId': _SERIALIZER.url('recommendation_id', recommendation_id, 'str'), 'name': _SERIALIZER.url('name', name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_request(resource_uri: str, recommendation_id: str, name: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-01-01'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{resourceUri}/providers/Microsoft.Advisor/recommendations/{recommendationId}/suppressions/{name}')
    path_format_arguments = {'resourceUri': _SERIALIZER.url('resource_uri', resource_uri, 'str'), 'recommendationId': _SERIALIZER.url('recommendation_id', recommendation_id, 'str'), 'name': _SERIALIZER.url('name', name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_uri: str, recommendation_id: str, name: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-01-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{resourceUri}/providers/Microsoft.Advisor/recommendations/{recommendationId}/suppressions/{name}')
    path_format_arguments = {'resourceUri': _SERIALIZER.url('resource_uri', resource_uri, 'str'), 'recommendationId': _SERIALIZER.url('recommendation_id', recommendation_id, 'str'), 'name': _SERIALIZER.url('name', name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_request(subscription_id: str, *, top: Optional[int]=None, skip_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-01-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/providers/Microsoft.Advisor/suppressions')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if top is not None:
        _params['$top'] = _SERIALIZER.query('top', top, 'int')
    if skip_token is not None:
        _params['$skipToken'] = _SERIALIZER.query('skip_token', skip_token, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class SuppressionsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.advisor.AdvisorManagementClient`'s
        :attr:`suppressions` attribute.
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
    def get(self, resource_uri: str, recommendation_id: str, name: str, **kwargs: Any) -> _models.SuppressionContract:
        if False:
            print('Hello World!')
        'Obtains the details of a suppression.\n\n        :param resource_uri: The fully qualified Azure Resource Manager identifier of the resource to\n         which the recommendation applies. Required.\n        :type resource_uri: str\n        :param recommendation_id: The recommendation ID. Required.\n        :type recommendation_id: str\n        :param name: The name of the suppression. Required.\n        :type name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SuppressionContract or the result of cls(response)\n        :rtype: ~azure.mgmt.advisor.models.SuppressionContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 409: ResourceExistsError, 304: ResourceNotModifiedError, 404: lambda response: ResourceNotFoundError(response=response, model=self._deserialize(_models.ArmErrorResponse, response), error_format=ARMErrorFormat)}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        request = build_get_request(resource_uri=resource_uri, recommendation_id=recommendation_id, name=name, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ArmErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SuppressionContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/{resourceUri}/providers/Microsoft.Advisor/recommendations/{recommendationId}/suppressions/{name}'}

    @overload
    def create(self, resource_uri: str, recommendation_id: str, name: str, suppression_contract: _models.SuppressionContract, *, content_type: str='application/json', **kwargs: Any) -> _models.SuppressionContract:
        if False:
            for i in range(10):
                print('nop')
        'Enables the snoozed or dismissed attribute of a recommendation. The snoozed or dismissed\n        attribute is referred to as a suppression. Use this API to create or update the snoozed or\n        dismissed status of a recommendation.\n\n        :param resource_uri: The fully qualified Azure Resource Manager identifier of the resource to\n         which the recommendation applies. Required.\n        :type resource_uri: str\n        :param recommendation_id: The recommendation ID. Required.\n        :type recommendation_id: str\n        :param name: The name of the suppression. Required.\n        :type name: str\n        :param suppression_contract: The snoozed or dismissed attribute; for example, the snooze\n         duration. Required.\n        :type suppression_contract: ~azure.mgmt.advisor.models.SuppressionContract\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SuppressionContract or the result of cls(response)\n        :rtype: ~azure.mgmt.advisor.models.SuppressionContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create(self, resource_uri: str, recommendation_id: str, name: str, suppression_contract: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.SuppressionContract:
        if False:
            i = 10
            return i + 15
        'Enables the snoozed or dismissed attribute of a recommendation. The snoozed or dismissed\n        attribute is referred to as a suppression. Use this API to create or update the snoozed or\n        dismissed status of a recommendation.\n\n        :param resource_uri: The fully qualified Azure Resource Manager identifier of the resource to\n         which the recommendation applies. Required.\n        :type resource_uri: str\n        :param recommendation_id: The recommendation ID. Required.\n        :type recommendation_id: str\n        :param name: The name of the suppression. Required.\n        :type name: str\n        :param suppression_contract: The snoozed or dismissed attribute; for example, the snooze\n         duration. Required.\n        :type suppression_contract: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SuppressionContract or the result of cls(response)\n        :rtype: ~azure.mgmt.advisor.models.SuppressionContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create(self, resource_uri: str, recommendation_id: str, name: str, suppression_contract: Union[_models.SuppressionContract, IO], **kwargs: Any) -> _models.SuppressionContract:
        if False:
            print('Hello World!')
        "Enables the snoozed or dismissed attribute of a recommendation. The snoozed or dismissed\n        attribute is referred to as a suppression. Use this API to create or update the snoozed or\n        dismissed status of a recommendation.\n\n        :param resource_uri: The fully qualified Azure Resource Manager identifier of the resource to\n         which the recommendation applies. Required.\n        :type resource_uri: str\n        :param recommendation_id: The recommendation ID. Required.\n        :type recommendation_id: str\n        :param name: The name of the suppression. Required.\n        :type name: str\n        :param suppression_contract: The snoozed or dismissed attribute; for example, the snooze\n         duration. Is either a model type or a IO type. Required.\n        :type suppression_contract: ~azure.mgmt.advisor.models.SuppressionContract or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SuppressionContract or the result of cls(response)\n        :rtype: ~azure.mgmt.advisor.models.SuppressionContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 409: ResourceExistsError, 304: ResourceNotModifiedError, 404: lambda response: ResourceNotFoundError(response=response, model=self._deserialize(_models.ArmErrorResponse, response), error_format=ARMErrorFormat)}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(suppression_contract, (IO, bytes)):
            _content = suppression_contract
        else:
            _json = self._serialize.body(suppression_contract, 'SuppressionContract')
        request = build_create_request(resource_uri=resource_uri, recommendation_id=recommendation_id, name=name, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ArmErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SuppressionContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create.metadata = {'url': '/{resourceUri}/providers/Microsoft.Advisor/recommendations/{recommendationId}/suppressions/{name}'}

    @distributed_trace
    def delete(self, resource_uri: str, recommendation_id: str, name: str, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Enables the activation of a snoozed or dismissed recommendation. The snoozed or dismissed\n        attribute of a recommendation is referred to as a suppression.\n\n        :param resource_uri: The fully qualified Azure Resource Manager identifier of the resource to\n         which the recommendation applies. Required.\n        :type resource_uri: str\n        :param recommendation_id: The recommendation ID. Required.\n        :type recommendation_id: str\n        :param name: The name of the suppression. Required.\n        :type name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        request = build_delete_request(resource_uri=resource_uri, recommendation_id=recommendation_id, name=name, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ArmErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/{resourceUri}/providers/Microsoft.Advisor/recommendations/{recommendationId}/suppressions/{name}'}

    @distributed_trace
    def list(self, top: Optional[int]=None, skip_token: Optional[str]=None, **kwargs: Any) -> Iterable['_models.SuppressionContract']:
        if False:
            i = 10
            return i + 15
        'Retrieves the list of snoozed or dismissed suppressions for a subscription. The snoozed or\n        dismissed attribute of a recommendation is referred to as a suppression.\n\n        :param top: The number of suppressions per page if a paged version of this API is being used.\n         Default value is None.\n        :type top: int\n        :param skip_token: The page-continuation token to use with a paged version of this API. Default\n         value is None.\n        :type skip_token: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SuppressionContract or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.advisor.models.SuppressionContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_request(subscription_id=self._config.subscription_id, top=top, skip_token=skip_token, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
                i = 10
                return i + 15
            deserialized = self._deserialize('SuppressionContractListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ArmErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.Advisor/suppressions'}