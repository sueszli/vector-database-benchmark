from io import IOBase
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
from .._vendor import _convert_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_by_database_account_request(resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'notebookWorkspaceName': _SERIALIZER.url('notebook_workspace_name', notebook_workspace_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_or_update_request(resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'notebookWorkspaceName': _SERIALIZER.url('notebook_workspace_name', notebook_workspace_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'notebookWorkspaceName': _SERIALIZER.url('notebook_workspace_name', notebook_workspace_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_connection_info_request(resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/listConnectionInfo')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'notebookWorkspaceName': _SERIALIZER.url('notebook_workspace_name', notebook_workspace_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_regenerate_auth_token_request(resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/regenerateAuthToken')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'notebookWorkspaceName': _SERIALIZER.url('notebook_workspace_name', notebook_workspace_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_start_request(resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/start')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'notebookWorkspaceName': _SERIALIZER.url('notebook_workspace_name', notebook_workspace_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class NotebookWorkspacesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.cosmosdb.CosmosDBManagementClient`'s
        :attr:`notebook_workspaces` attribute.
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
    def list_by_database_account(self, resource_group_name: str, account_name: str, **kwargs: Any) -> Iterable['_models.NotebookWorkspace']:
        if False:
            while True:
                i = 10
        'Gets the notebook workspace resources of an existing Cosmos DB account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either NotebookWorkspace or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.NotebookWorkspace]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.NotebookWorkspaceListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                request = build_list_by_database_account_request(resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_by_database_account.metadata['url'], headers=_headers, params=_params)
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
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('NotebookWorkspaceListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_database_account.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces'}

    @distributed_trace
    def get(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> _models.NotebookWorkspace:
        if False:
            i = 10
            return i + 15
        'Gets the notebook workspace for a Cosmos DB account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: NotebookWorkspace or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.NotebookWorkspace\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.NotebookWorkspace] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('NotebookWorkspace', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}'}

    def _create_or_update_initial(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], notebook_create_update_parameters: Union[_models.NotebookWorkspaceCreateUpdateParameters, IO], **kwargs: Any) -> _models.NotebookWorkspace:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.NotebookWorkspace] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(notebook_create_update_parameters, (IOBase, bytes)):
            _content = notebook_create_update_parameters
        else:
            _json = self._serialize.body(notebook_create_update_parameters, 'NotebookWorkspaceCreateUpdateParameters')
        request = build_create_or_update_request(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_or_update_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('NotebookWorkspace', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_or_update_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}'}

    @overload
    def begin_create_or_update(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], notebook_create_update_parameters: _models.NotebookWorkspaceCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.NotebookWorkspace]:
        if False:
            for i in range(10):
                print('nop')
        'Creates the notebook workspace for a Cosmos DB account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :param notebook_create_update_parameters: The notebook workspace to create for the current\n         database account. Required.\n        :type notebook_create_update_parameters:\n         ~azure.mgmt.cosmosdb.models.NotebookWorkspaceCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either NotebookWorkspace or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.NotebookWorkspace]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_or_update(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], notebook_create_update_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.NotebookWorkspace]:
        if False:
            for i in range(10):
                print('nop')
        'Creates the notebook workspace for a Cosmos DB account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :param notebook_create_update_parameters: The notebook workspace to create for the current\n         database account. Required.\n        :type notebook_create_update_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either NotebookWorkspace or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.NotebookWorkspace]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_or_update(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], notebook_create_update_parameters: Union[_models.NotebookWorkspaceCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.NotebookWorkspace]:
        if False:
            i = 10
            return i + 15
        'Creates the notebook workspace for a Cosmos DB account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :param notebook_create_update_parameters: The notebook workspace to create for the current\n         database account. Is either a NotebookWorkspaceCreateUpdateParameters type or a IO type.\n         Required.\n        :type notebook_create_update_parameters:\n         ~azure.mgmt.cosmosdb.models.NotebookWorkspaceCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: \'application/json\'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either NotebookWorkspace or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.NotebookWorkspace]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.NotebookWorkspace] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_or_update_initial(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, notebook_create_update_parameters=notebook_create_update_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('NotebookWorkspace', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}'}

    def _delete_initial(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _delete_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}'}

    @distributed_trace
    def begin_delete(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> LROPoller[None]:
        if False:
            print('Hello World!')
        'Deletes the notebook workspace for a Cosmos DB account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_initial(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}'}

    @distributed_trace
    def list_connection_info(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> _models.NotebookWorkspaceConnectionInfoResult:
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the connection info for the notebook workspace.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: NotebookWorkspaceConnectionInfoResult or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.NotebookWorkspaceConnectionInfoResult\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.NotebookWorkspaceConnectionInfoResult] = kwargs.pop('cls', None)
        request = build_list_connection_info_request(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_connection_info.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('NotebookWorkspaceConnectionInfoResult', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_connection_info.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/listConnectionInfo'}

    def _regenerate_auth_token_initial(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_regenerate_auth_token_request(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._regenerate_auth_token_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _regenerate_auth_token_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/regenerateAuthToken'}

    @distributed_trace
    def begin_regenerate_auth_token(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> LROPoller[None]:
        if False:
            return 10
        'Regenerates the auth token for the notebook workspace.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._regenerate_auth_token_initial(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_regenerate_auth_token.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/regenerateAuthToken'}

    def _start_initial(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> None:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_start_request(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._start_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _start_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/start'}

    @distributed_trace
    def begin_start(self, resource_group_name: str, account_name: str, notebook_workspace_name: Union[str, _models.NotebookWorkspaceName], **kwargs: Any) -> LROPoller[None]:
        if False:
            for i in range(10):
                print('nop')
        'Starts the notebook workspace.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param notebook_workspace_name: The name of the notebook workspace resource. "default"\n         Required.\n        :type notebook_workspace_name: str or ~azure.mgmt.cosmosdb.models.NotebookWorkspaceName\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._start_initial(resource_group_name=resource_group_name, account_name=account_name, notebook_workspace_name=notebook_workspace_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_start.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/notebookWorkspaces/{notebookWorkspaceName}/start'}