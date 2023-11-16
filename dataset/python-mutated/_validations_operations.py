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

def build_validate_organization_request(resource_group_name: str, organization_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2021-12-01'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Confluent/validations/{organizationName}/orgvalidate')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'organizationName': _SERIALIZER.url('organization_name', organization_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class ValidationsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.confluent.ConfluentManagementClient`'s
        :attr:`validations` attribute.
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
    def validate_organization(self, resource_group_name: str, organization_name: str, body: _models.OrganizationResource, *, content_type: str='application/json', **kwargs: Any) -> _models.OrganizationResource:
        if False:
            for i in range(10):
                print('nop')
        'Organization Validate proxy resource.\n\n        Organization Validate proxy resource.\n\n        :param resource_group_name: Resource group name. Required.\n        :type resource_group_name: str\n        :param organization_name: Organization resource name. Required.\n        :type organization_name: str\n        :param body: Organization resource model. Required.\n        :type body: ~azure.mgmt.confluent.models.OrganizationResource\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: OrganizationResource or the result of cls(response)\n        :rtype: ~azure.mgmt.confluent.models.OrganizationResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def validate_organization(self, resource_group_name: str, organization_name: str, body: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.OrganizationResource:
        if False:
            i = 10
            return i + 15
        'Organization Validate proxy resource.\n\n        Organization Validate proxy resource.\n\n        :param resource_group_name: Resource group name. Required.\n        :type resource_group_name: str\n        :param organization_name: Organization resource name. Required.\n        :type organization_name: str\n        :param body: Organization resource model. Required.\n        :type body: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: OrganizationResource or the result of cls(response)\n        :rtype: ~azure.mgmt.confluent.models.OrganizationResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def validate_organization(self, resource_group_name: str, organization_name: str, body: Union[_models.OrganizationResource, IO], **kwargs: Any) -> _models.OrganizationResource:
        if False:
            i = 10
            return i + 15
        "Organization Validate proxy resource.\n\n        Organization Validate proxy resource.\n\n        :param resource_group_name: Resource group name. Required.\n        :type resource_group_name: str\n        :param organization_name: Organization resource name. Required.\n        :type organization_name: str\n        :param body: Organization resource model. Is either a model type or a IO type. Required.\n        :type body: ~azure.mgmt.confluent.models.OrganizationResource or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: OrganizationResource or the result of cls(response)\n        :rtype: ~azure.mgmt.confluent.models.OrganizationResource\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
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
        if isinstance(body, (IO, bytes)):
            _content = body
        else:
            _json = self._serialize.body(body, 'OrganizationResource')
        request = build_validate_organization_request(resource_group_name=resource_group_name, organization_name=organization_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.validate_organization.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ResourceProviderDefaultErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('OrganizationResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    validate_organization.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Confluent/validations/{organizationName}/orgvalidate'}