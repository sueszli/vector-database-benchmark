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

def build_create_or_update_request(scope: str, association_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2018-09-01-preview'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'associationName': _SERIALIZER.url('association_name', association_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(scope: str, association_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2018-09-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'associationName': _SERIALIZER.url('association_name', association_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(scope: str, association_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2018-09-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'associationName': _SERIALIZER.url('association_name', association_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_all_request(scope: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2018-09-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.CustomProviders/associations')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True)}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class AssociationsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.customproviders.Customproviders`'s
        :attr:`associations` attribute.
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

    def _create_or_update_initial(self, scope: str, association_name: str, association: Union[_models.Association, IO], **kwargs: Any) -> _models.Association:
        if False:
            return 10
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
        if isinstance(association, (IO, bytes)):
            _content = association
        else:
            _json = self._serialize.body(association, 'Association')
        request = build_create_or_update_request(scope=scope, association_name=association_name, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_or_update_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if response.status_code == 200:
            deserialized = self._deserialize('Association', pipeline_response)
        if response.status_code == 201:
            deserialized = self._deserialize('Association', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_or_update_initial.metadata = {'url': '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}'}

    @overload
    def begin_create_or_update(self, scope: str, association_name: str, association: _models.Association, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.Association]:
        if False:
            i = 10
            return i + 15
        'Create or update an association.\n\n        :param scope: The scope of the association. The scope can be any valid REST resource instance.\n         For example, use\n         \'/subscriptions/{subscription-id}/resourceGroups/{resource-group-name}/providers/Microsoft.Compute/virtualMachines/{vm-name}\'\n         for a virtual machine resource. Required.\n        :type scope: str\n        :param association_name: The name of the association. Required.\n        :type association_name: str\n        :param association: The parameters required to create or update an association. Required.\n        :type association: ~azure.mgmt.customproviders.models.Association\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Association or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.customproviders.models.Association]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_or_update(self, scope: str, association_name: str, association: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.Association]:
        if False:
            while True:
                i = 10
        'Create or update an association.\n\n        :param scope: The scope of the association. The scope can be any valid REST resource instance.\n         For example, use\n         \'/subscriptions/{subscription-id}/resourceGroups/{resource-group-name}/providers/Microsoft.Compute/virtualMachines/{vm-name}\'\n         for a virtual machine resource. Required.\n        :type scope: str\n        :param association_name: The name of the association. Required.\n        :type association_name: str\n        :param association: The parameters required to create or update an association. Required.\n        :type association: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Association or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.customproviders.models.Association]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_or_update(self, scope: str, association_name: str, association: Union[_models.Association, IO], **kwargs: Any) -> LROPoller[_models.Association]:
        if False:
            return 10
        "Create or update an association.\n\n        :param scope: The scope of the association. The scope can be any valid REST resource instance.\n         For example, use\n         '/subscriptions/{subscription-id}/resourceGroups/{resource-group-name}/providers/Microsoft.Compute/virtualMachines/{vm-name}'\n         for a virtual machine resource. Required.\n        :type scope: str\n        :param association_name: The name of the association. Required.\n        :type association_name: str\n        :param association: The parameters required to create or update an association. Is either a\n         model type or a IO type. Required.\n        :type association: ~azure.mgmt.customproviders.models.Association or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Association or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.customproviders.models.Association]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        polling = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_or_update_initial(scope=scope, association_name=association_name, association=association, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('Association', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_or_update.metadata = {'url': '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}'}

    def _delete_initial(self, scope: str, association_name: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        request = build_delete_request(scope=scope, association_name=association_name, api_version=api_version, template_url=self._delete_initial.metadata['url'], headers=_headers, params=_params)
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
    _delete_initial.metadata = {'url': '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}'}

    @distributed_trace
    def begin_delete(self, scope: str, association_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            for i in range(10):
                print('nop')
        'Delete an association.\n\n        :param scope: The scope of the association. Required.\n        :type scope: str\n        :param association_name: The name of the association. Required.\n        :type association_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        polling = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_initial(scope=scope, association_name=association_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete.metadata = {'url': '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}'}

    @distributed_trace
    def get(self, scope: str, association_name: str, **kwargs: Any) -> _models.Association:
        if False:
            print('Hello World!')
        'Get an association.\n\n        :param scope: The scope of the association. Required.\n        :type scope: str\n        :param association_name: The name of the association. Required.\n        :type association_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Association or the result of cls(response)\n        :rtype: ~azure.mgmt.customproviders.models.Association\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        request = build_get_request(scope=scope, association_name=association_name, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Association', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/{scope}/providers/Microsoft.CustomProviders/associations/{associationName}'}

    @distributed_trace
    def list_all(self, scope: str, **kwargs: Any) -> Iterable['_models.Association']:
        if False:
            print('Hello World!')
        'Gets all association for the given scope.\n\n        :param scope: The scope of the association. Required.\n        :type scope: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Association or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.customproviders.models.Association]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                request = build_list_all_request(scope=scope, api_version=api_version, template_url=self.list_all.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('AssociationsList', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_all.metadata = {'url': '/{scope}/providers/Microsoft.CustomProviders/associations'}