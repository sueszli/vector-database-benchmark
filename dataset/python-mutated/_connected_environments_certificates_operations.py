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
from .._serialization import Serializer
from .._vendor import ContainerAppsAPIClientMixinABC, _convert_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_request(resource_group_name: str, connected_environment_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'connectedEnvironmentName': _SERIALIZER.url('connected_environment_name', connected_environment_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, connected_environment_name: str, certificate_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'connectedEnvironmentName': _SERIALIZER.url('connected_environment_name', connected_environment_name, 'str'), 'certificateName': _SERIALIZER.url('certificate_name', certificate_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, connected_environment_name: str, certificate_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-05-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'connectedEnvironmentName': _SERIALIZER.url('connected_environment_name', connected_environment_name, 'str'), 'certificateName': _SERIALIZER.url('certificate_name', certificate_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, connected_environment_name: str, certificate_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'connectedEnvironmentName': _SERIALIZER.url('connected_environment_name', connected_environment_name, 'str'), 'certificateName': _SERIALIZER.url('certificate_name', certificate_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, connected_environment_name: str, certificate_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-05-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'connectedEnvironmentName': _SERIALIZER.url('connected_environment_name', connected_environment_name, 'str'), 'certificateName': _SERIALIZER.url('certificate_name', certificate_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

class ConnectedEnvironmentsCertificatesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.appcontainers.ContainerAppsAPIClient`'s
        :attr:`connected_environments_certificates` attribute.
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
    def list(self, resource_group_name: str, connected_environment_name: str, **kwargs: Any) -> Iterable['_models.Certificate']:
        if False:
            print('Hello World!')
        'Get the Certificates in a given connected environment.\n\n        Get the Certificates in a given connected environment.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Certificate or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.appcontainers.models.Certificate]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.CertificateCollection] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_request(resource_group_name=resource_group_name, connected_environment_name=connected_environment_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('CertificateCollection', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.DefaultErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates'}

    @distributed_trace
    def get(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, **kwargs: Any) -> _models.Certificate:
        if False:
            i = 10
            return i + 15
        'Get the specified Certificate.\n\n        Get the specified Certificate.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Certificate or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.Certificate\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.Certificate] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, connected_environment_name=connected_environment_name, certificate_name=certificate_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.DefaultErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Certificate', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}'}

    @overload
    def create_or_update(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, certificate_envelope: Optional[_models.Certificate]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Certificate:
        if False:
            i = 10
            return i + 15
        'Create or Update a Certificate.\n\n        Create or Update a Certificate.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :param certificate_envelope: Certificate to be created or updated. Default value is None.\n        :type certificate_envelope: ~azure.mgmt.appcontainers.models.Certificate\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Certificate or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.Certificate\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def create_or_update(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, certificate_envelope: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Certificate:
        if False:
            return 10
        'Create or Update a Certificate.\n\n        Create or Update a Certificate.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :param certificate_envelope: Certificate to be created or updated. Default value is None.\n        :type certificate_envelope: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Certificate or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.Certificate\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def create_or_update(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, certificate_envelope: Optional[Union[_models.Certificate, IO]]=None, **kwargs: Any) -> _models.Certificate:
        if False:
            for i in range(10):
                print('nop')
        "Create or Update a Certificate.\n\n        Create or Update a Certificate.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :param certificate_envelope: Certificate to be created or updated. Is either a Certificate type\n         or a IO type. Default value is None.\n        :type certificate_envelope: ~azure.mgmt.appcontainers.models.Certificate or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Certificate or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.Certificate\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Certificate] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(certificate_envelope, (IOBase, bytes)):
            _content = certificate_envelope
        elif certificate_envelope is not None:
            _json = self._serialize.body(certificate_envelope, 'Certificate')
        else:
            _json = None
        request = build_create_or_update_request(resource_group_name=resource_group_name, connected_environment_name=connected_environment_name, certificate_name=certificate_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.create_or_update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.DefaultErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Certificate', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}'}

    @distributed_trace
    def delete(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Deletes the specified Certificate.\n\n        Deletes the specified Certificate.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, connected_environment_name=connected_environment_name, certificate_name=certificate_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.DefaultErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}'}

    @overload
    def update(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, certificate_envelope: _models.CertificatePatch, *, content_type: str='application/json', **kwargs: Any) -> _models.Certificate:
        if False:
            while True:
                i = 10
        'Update properties of a certificate.\n\n        Patches a certificate. Currently only patching of tags is supported.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :param certificate_envelope: Properties of a certificate that need to be updated. Required.\n        :type certificate_envelope: ~azure.mgmt.appcontainers.models.CertificatePatch\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Certificate or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.Certificate\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, certificate_envelope: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.Certificate:
        if False:
            i = 10
            return i + 15
        'Update properties of a certificate.\n\n        Patches a certificate. Currently only patching of tags is supported.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :param certificate_envelope: Properties of a certificate that need to be updated. Required.\n        :type certificate_envelope: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Certificate or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.Certificate\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, connected_environment_name: str, certificate_name: str, certificate_envelope: Union[_models.CertificatePatch, IO], **kwargs: Any) -> _models.Certificate:
        if False:
            print('Hello World!')
        "Update properties of a certificate.\n\n        Patches a certificate. Currently only patching of tags is supported.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param connected_environment_name: Name of the Connected Environment. Required.\n        :type connected_environment_name: str\n        :param certificate_name: Name of the Certificate. Required.\n        :type certificate_name: str\n        :param certificate_envelope: Properties of a certificate that need to be updated. Is either a\n         CertificatePatch type or a IO type. Required.\n        :type certificate_envelope: ~azure.mgmt.appcontainers.models.CertificatePatch or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Certificate or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.Certificate\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Certificate] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(certificate_envelope, (IOBase, bytes)):
            _content = certificate_envelope
        else:
            _json = self._serialize.body(certificate_envelope, 'CertificatePatch')
        request = build_update_request(resource_group_name=resource_group_name, connected_environment_name=connected_environment_name, certificate_name=certificate_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.DefaultErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Certificate', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/connectedEnvironments/{connectedEnvironmentName}/certificates/{certificateName}'}