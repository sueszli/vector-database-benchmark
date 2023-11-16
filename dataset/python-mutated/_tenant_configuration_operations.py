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
from .._vendor import ApiManagementClientMixinABC, _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_deploy_request(resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/deploy')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'configurationName': _SERIALIZER.url('configuration_name', configuration_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_save_request(resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/save')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'configurationName': _SERIALIZER.url('configuration_name', configuration_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_validate_request(resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/validate')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'configurationName': _SERIALIZER.url('configuration_name', configuration_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_sync_state_request(resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-08-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/syncState')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'serviceName': _SERIALIZER.url('service_name', service_name, 'str', max_length=50, min_length=1, pattern='^[a-zA-Z](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'configurationName': _SERIALIZER.url('configuration_name', configuration_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class TenantConfigurationOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.apimanagement.ApiManagementClient`'s
        :attr:`tenant_configuration` attribute.
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

    def _deploy_initial(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: Union[_models.DeployConfigurationParameters, IO], **kwargs: Any) -> Optional[_models.OperationResultContract]:
        if False:
            print('Hello World!')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.OperationResultContract]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'DeployConfigurationParameters')
        request = build_deploy_request(resource_group_name=resource_group_name, service_name=service_name, configuration_name=configuration_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._deploy_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationResultContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _deploy_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/deploy'}

    @overload
    def begin_deploy(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: _models.DeployConfigurationParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            i = 10
            return i + 15
        'This operation applies changes from the specified Git branch to the configuration database.\n        This is a long running operation and could take several minutes to complete.\n\n        .. seealso::\n           -\n        https://azure.microsoft.com/en-us/documentation/articles/api-management-configuration-repository-git/#to-deploy-any-service-configuration-changes-to-the-api-management-service-instance\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Deploy Configuration parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.DeployConfigurationParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_deploy(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            for i in range(10):
                print('nop')
        'This operation applies changes from the specified Git branch to the configuration database.\n        This is a long running operation and could take several minutes to complete.\n\n        .. seealso::\n           -\n        https://azure.microsoft.com/en-us/documentation/articles/api-management-configuration-repository-git/#to-deploy-any-service-configuration-changes-to-the-api-management-service-instance\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Deploy Configuration parameters. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_deploy(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: Union[_models.DeployConfigurationParameters, IO], **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            return 10
        'This operation applies changes from the specified Git branch to the configuration database.\n        This is a long running operation and could take several minutes to complete.\n\n        .. seealso::\n           -\n        https://azure.microsoft.com/en-us/documentation/articles/api-management-configuration-repository-git/#to-deploy-any-service-configuration-changes-to-the-api-management-service-instance\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Deploy Configuration parameters. Is either a DeployConfigurationParameters\n         type or a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.DeployConfigurationParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: \'application/json\'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.OperationResultContract] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._deploy_initial(resource_group_name=resource_group_name, service_name=service_name, configuration_name=configuration_name, parameters=parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('OperationResultContract', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'location'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_deploy.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/deploy'}

    def _save_initial(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: Union[_models.SaveConfigurationParameter, IO], **kwargs: Any) -> Optional[_models.OperationResultContract]:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.OperationResultContract]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'SaveConfigurationParameter')
        request = build_save_request(resource_group_name=resource_group_name, service_name=service_name, configuration_name=configuration_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._save_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationResultContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _save_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/save'}

    @overload
    def begin_save(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: _models.SaveConfigurationParameter, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            i = 10
            return i + 15
        'This operation creates a commit with the current configuration snapshot to the specified branch\n        in the repository. This is a long running operation and could take several minutes to complete.\n\n        .. seealso::\n           -\n        https://azure.microsoft.com/en-us/documentation/articles/api-management-configuration-repository-git/#to-save-the-service-configuration-to-the-git-repository\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Save Configuration parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.SaveConfigurationParameter\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_save(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            while True:
                i = 10
        'This operation creates a commit with the current configuration snapshot to the specified branch\n        in the repository. This is a long running operation and could take several minutes to complete.\n\n        .. seealso::\n           -\n        https://azure.microsoft.com/en-us/documentation/articles/api-management-configuration-repository-git/#to-save-the-service-configuration-to-the-git-repository\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Save Configuration parameters. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_save(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: Union[_models.SaveConfigurationParameter, IO], **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            return 10
        'This operation creates a commit with the current configuration snapshot to the specified branch\n        in the repository. This is a long running operation and could take several minutes to complete.\n\n        .. seealso::\n           -\n        https://azure.microsoft.com/en-us/documentation/articles/api-management-configuration-repository-git/#to-save-the-service-configuration-to-the-git-repository\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Save Configuration parameters. Is either a SaveConfigurationParameter type\n         or a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.SaveConfigurationParameter or IO\n        :keyword content_type: Body Parameter content-type. Known values are: \'application/json\'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.OperationResultContract] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._save_initial(resource_group_name=resource_group_name, service_name=service_name, configuration_name=configuration_name, parameters=parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('OperationResultContract', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'location'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_save.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/save'}

    def _validate_initial(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: Union[_models.DeployConfigurationParameters, IO], **kwargs: Any) -> Optional[_models.OperationResultContract]:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.OperationResultContract]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'DeployConfigurationParameters')
        request = build_validate_request(resource_group_name=resource_group_name, service_name=service_name, configuration_name=configuration_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._validate_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationResultContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _validate_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/validate'}

    @overload
    def begin_validate(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: _models.DeployConfigurationParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            while True:
                i = 10
        'This operation validates the changes in the specified Git branch. This is a long running\n        operation and could take several minutes to complete.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Validate Configuration parameters. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.DeployConfigurationParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_validate(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            while True:
                i = 10
        'This operation validates the changes in the specified Git branch. This is a long running\n        operation and could take several minutes to complete.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Validate Configuration parameters. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_validate(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], parameters: Union[_models.DeployConfigurationParameters, IO], **kwargs: Any) -> LROPoller[_models.OperationResultContract]:
        if False:
            i = 10
            return i + 15
        'This operation validates the changes in the specified Git branch. This is a long running\n        operation and could take several minutes to complete.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :param parameters: Validate Configuration parameters. Is either a DeployConfigurationParameters\n         type or a IO type. Required.\n        :type parameters: ~azure.mgmt.apimanagement.models.DeployConfigurationParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: \'application/json\'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either OperationResultContract or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.apimanagement.models.OperationResultContract]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.OperationResultContract] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._validate_initial(resource_group_name=resource_group_name, service_name=service_name, configuration_name=configuration_name, parameters=parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('OperationResultContract', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'location'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_validate.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/validate'}

    @distributed_trace
    def get_sync_state(self, resource_group_name: str, service_name: str, configuration_name: Union[str, _models.ConfigurationIdName], **kwargs: Any) -> _models.TenantConfigurationSyncStateContract:
        if False:
            print('Hello World!')
        'Gets the status of the most recent synchronization between the configuration database and the\n        Git repository.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param service_name: The name of the API Management service. Required.\n        :type service_name: str\n        :param configuration_name: The identifier of the Git Configuration Operation. "configuration"\n         Required.\n        :type configuration_name: str or ~azure.mgmt.apimanagement.models.ConfigurationIdName\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: TenantConfigurationSyncStateContract or the result of cls(response)\n        :rtype: ~azure.mgmt.apimanagement.models.TenantConfigurationSyncStateContract\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.TenantConfigurationSyncStateContract] = kwargs.pop('cls', None)
        request = build_get_sync_state_request(resource_group_name=resource_group_name, service_name=service_name, configuration_name=configuration_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sync_state.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('TenantConfigurationSyncStateContract', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sync_state.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ApiManagement/service/{serviceName}/tenant/{configurationName}/syncState'}