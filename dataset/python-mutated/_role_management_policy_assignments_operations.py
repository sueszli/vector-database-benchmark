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
from ..._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(scope: str, role_management_policy_assignment_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2020-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments/{roleManagementPolicyAssignmentName}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'roleManagementPolicyAssignmentName': _SERIALIZER.url('role_management_policy_assignment_name', role_management_policy_assignment_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_request(scope: str, role_management_policy_assignment_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2020-10-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments/{roleManagementPolicyAssignmentName}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'roleManagementPolicyAssignmentName': _SERIALIZER.url('role_management_policy_assignment_name', role_management_policy_assignment_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(scope: str, role_management_policy_assignment_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2020-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments/{roleManagementPolicyAssignmentName}')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True), 'roleManagementPolicyAssignmentName': _SERIALIZER.url('role_management_policy_assignment_name', role_management_policy_assignment_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_for_scope_request(scope: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2020-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments')
    path_format_arguments = {'scope': _SERIALIZER.url('scope', scope, 'str', skip_quote=True)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class RoleManagementPolicyAssignmentsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.authorization.v2020_10_01.AuthorizationManagementClient`'s
        :attr:`role_management_policy_assignments` attribute.
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
        self._api_version = input_args.pop(0) if input_args else kwargs.pop('api_version')

    @distributed_trace
    def get(self, scope: str, role_management_policy_assignment_name: str, **kwargs: Any) -> _models.RoleManagementPolicyAssignment:
        if False:
            while True:
                i = 10
        'Get the specified role management policy assignment for a resource scope.\n\n        :param scope: The scope of the role management policy. Required.\n        :type scope: str\n        :param role_management_policy_assignment_name: The name of format {guid_guid} the role\n         management policy assignment to get. Required.\n        :type role_management_policy_assignment_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleManagementPolicyAssignment or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2020_10_01.models.RoleManagementPolicyAssignment\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2020-10-01'))
        cls: ClsType[_models.RoleManagementPolicyAssignment] = kwargs.pop('cls', None)
        request = build_get_request(scope=scope, role_management_policy_assignment_name=role_management_policy_assignment_name, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('RoleManagementPolicyAssignment', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments/{roleManagementPolicyAssignmentName}'}

    @overload
    def create(self, scope: str, role_management_policy_assignment_name: str, parameters: _models.RoleManagementPolicyAssignment, *, content_type: str='application/json', **kwargs: Any) -> _models.RoleManagementPolicyAssignment:
        if False:
            i = 10
            return i + 15
        'Create a role management policy assignment.\n\n        :param scope: The scope of the role management policy assignment to upsert. Required.\n        :type scope: str\n        :param role_management_policy_assignment_name: The name of format {guid_guid} the role\n         management policy assignment to upsert. Required.\n        :type role_management_policy_assignment_name: str\n        :param parameters: Parameters for the role management policy assignment. Required.\n        :type parameters: ~azure.mgmt.authorization.v2020_10_01.models.RoleManagementPolicyAssignment\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleManagementPolicyAssignment or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2020_10_01.models.RoleManagementPolicyAssignment\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create(self, scope: str, role_management_policy_assignment_name: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.RoleManagementPolicyAssignment:
        if False:
            i = 10
            return i + 15
        'Create a role management policy assignment.\n\n        :param scope: The scope of the role management policy assignment to upsert. Required.\n        :type scope: str\n        :param role_management_policy_assignment_name: The name of format {guid_guid} the role\n         management policy assignment to upsert. Required.\n        :type role_management_policy_assignment_name: str\n        :param parameters: Parameters for the role management policy assignment. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleManagementPolicyAssignment or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2020_10_01.models.RoleManagementPolicyAssignment\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create(self, scope: str, role_management_policy_assignment_name: str, parameters: Union[_models.RoleManagementPolicyAssignment, IO], **kwargs: Any) -> _models.RoleManagementPolicyAssignment:
        if False:
            i = 10
            return i + 15
        "Create a role management policy assignment.\n\n        :param scope: The scope of the role management policy assignment to upsert. Required.\n        :type scope: str\n        :param role_management_policy_assignment_name: The name of format {guid_guid} the role\n         management policy assignment to upsert. Required.\n        :type role_management_policy_assignment_name: str\n        :param parameters: Parameters for the role management policy assignment. Is either a\n         RoleManagementPolicyAssignment type or a IO type. Required.\n        :type parameters: ~azure.mgmt.authorization.v2020_10_01.models.RoleManagementPolicyAssignment\n         or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: RoleManagementPolicyAssignment or the result of cls(response)\n        :rtype: ~azure.mgmt.authorization.v2020_10_01.models.RoleManagementPolicyAssignment\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2020-10-01'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.RoleManagementPolicyAssignment] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IOBase, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'RoleManagementPolicyAssignment')
        request = build_create_request(scope=scope, role_management_policy_assignment_name=role_management_policy_assignment_name, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('RoleManagementPolicyAssignment', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments/{roleManagementPolicyAssignmentName}'}

    @distributed_trace
    def delete(self, scope: str, role_management_policy_assignment_name: str, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Delete a role management policy assignment.\n\n        :param scope: The scope of the role management policy assignment to delete. Required.\n        :type scope: str\n        :param role_management_policy_assignment_name: The name of format {guid_guid} the role\n         management policy assignment to delete. Required.\n        :type role_management_policy_assignment_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2020-10-01'))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(scope=scope, role_management_policy_assignment_name=role_management_policy_assignment_name, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
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
    delete.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments/{roleManagementPolicyAssignmentName}'}

    @distributed_trace
    def list_for_scope(self, scope: str, **kwargs: Any) -> Iterable['_models.RoleManagementPolicyAssignment']:
        if False:
            return 10
        'Gets role management assignment policies for a resource scope.\n\n        :param scope: The scope of the role management policy. Required.\n        :type scope: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either RoleManagementPolicyAssignment or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.authorization.v2020_10_01.models.RoleManagementPolicyAssignment]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._api_version or '2020-10-01'))
        cls: ClsType[_models.RoleManagementPolicyAssignmentListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_for_scope_request(scope=scope, api_version=api_version, template_url=self.list_for_scope.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('RoleManagementPolicyAssignmentListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_for_scope.metadata = {'url': '/{scope}/providers/Microsoft.Authorization/roleManagementPolicyAssignments'}