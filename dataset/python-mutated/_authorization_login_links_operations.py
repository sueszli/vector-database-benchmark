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

def build_post_request(resource_group_name: str, service_name: str, authorization_provider_id: str, authorization_id: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationProviders/{authorizationProviderId}/authorizations/{authorizationId}/getLoginLinks')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'authorizationProviderId': _SERIALIZER.url('authorization_provider_id', authorization_provider_id, 'str', max_length=256, min_length=1, pattern='^[^*#&+:<>?]+$'), 'authorizationId': _SERIALIZER.url('authorization_id', authorization_id, 'str', max_length=256, min_length=1, pattern='^[^*#&+:<>?]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1)}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class AuthorizationLoginLinksOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.apimanagement.ApiManagementClient`'s
        :attr:`authorization_login_links` attribute.
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
    def post(self, resource_group_name: str, service_name: str, authorization_provider_id: str, authorization_id: str, parameters: _models.AuthorizationLoginRequestContract, *, content_type: str='application/json', **kwargs: Any) -> _models.AuthorizationLoginResponseContract:
        if False:
            print('Hello World!')
        'Gets authorization login links.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authorization_provider_id: Identifier of the authorization provider. Required.\n        :type authorization_provider_id: str\n        :param authorization_id: Identifier of the authorization. Required.\n        :type authorization_id: str\n        :param parameters: Create parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.AuthorizationLoginRequestContract\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationLoginResponseContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationLoginResponseContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def post(self, resource_group_name: str, service_name: str, authorization_provider_id: str, authorization_id: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.AuthorizationLoginResponseContract:
        if False:
            while True:
                i = 10
        'Gets authorization login links.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authorization_provider_id: Identifier of the authorization provider. Required.\n        :type authorization_provider_id: str\n        :param authorization_id: Identifier of the authorization. Required.\n        :type authorization_id: str\n        :param parameters: Create parameters. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationLoginResponseContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationLoginResponseContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def post(self, resource_group_name: str, service_name: str, authorization_provider_id: str, authorization_id: str, parameters: Union[_models.AuthorizationLoginRequestContract, IO], **kwargs: Any) -> _models.AuthorizationLoginResponseContract:
        if False:
            print('Hello World!')
        "Gets authorization login links.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param authorization_provider_id: Identifier of the authorization provider. Required.\n        :type authorization_provider_id: str\n        :param authorization_id: Identifier of the authorization. Required.\n        :type authorization_id: str\n        :param parameters: Create parameters. Is either a AuthorizationLoginRequestContract type or a\n         IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.AuthorizationLoginRequestContract or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AuthorizationLoginResponseContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.AuthorizationLoginResponseContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.AuthorizationLoginResponseContract] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'AuthorizationLoginRequestContract')
        request = build_post_request(resource_group_name=resource_group_name, service_name=service_name, authorization_provider_id=authorization_provider_id, authorization_id=authorization_id, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.post.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        deserialized = self._deserialize('AuthorizationLoginResponseContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    post.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/authorizationProviders/{authorizationProviderId}/authorizations/{authorizationId}/getLoginLinks'}