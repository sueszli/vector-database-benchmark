from typing import TYPE_CHECKING
import warnings
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from azure.core.polling import LROPoller, NoPolling, PollingMethod
from azure.mgmt.core.exceptions import ARMErrorFormat
from azure.mgmt.core.polling.arm_polling import ARMPolling
from .. import models as _models
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar, Union
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

class FileServersOperations(object):
    """FileServersOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~batch_ai.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            print('Hello World!')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    def _create_initial(self, resource_group_name, workspace_name, file_server_name, parameters, **kwargs):
        if False:
            print('Hello World!')
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2018-05-01'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self._create_initial.metadata['url']
        path_format_arguments = {'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', pattern='^[-\\w\\._]+$'), 'workspaceName': self._serialize.url('workspace_name', workspace_name, 'str', max_length=64, min_length=1, pattern='^[-\\w_]+$'), 'fileServerName': self._serialize.url('file_server_name', file_server_name, 'str', max_length=64, min_length=1, pattern='^[-\\w_]+$'), 'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(parameters, 'FileServerCreateParameters')
        body_content_kwargs['content'] = body_content
        request = self._client.put(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('FileServer', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BatchAI/workspaces/{workspaceName}/fileServers/{fileServerName}'}

    def begin_create(self, resource_group_name, workspace_name, file_server_name, parameters, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Creates a File Server in the given workspace.\n\n        :param resource_group_name: Name of the resource group to which the resource belongs.\n        :type resource_group_name: str\n        :param workspace_name: The name of the workspace. Workspace names can only contain a\n         combination of alphanumeric characters along with dash (-) and underscore (_). The name must be\n         from 1 through 64 characters long.\n        :type workspace_name: str\n        :param file_server_name: The name of the file server within the specified resource group. File\n         server names can only contain a combination of alphanumeric characters along with dash (-) and\n         underscore (_). The name must be from 1 through 64 characters long.\n        :type file_server_name: str\n        :param parameters: The parameters to provide for File Server creation.\n        :type parameters: ~batch_ai.models.FileServerCreateParameters\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling.\n         Pass in False for this operation to not poll, or pass in your own initialized polling object for a personal polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no Retry-After header is present.\n        :return: An instance of LROPoller that returns either FileServer or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~batch_ai.models.FileServer]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        polling = kwargs.pop('polling', True)
        cls = kwargs.pop('cls', None)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_initial(resource_group_name=resource_group_name, workspace_name=workspace_name, file_server_name=file_server_name, parameters=parameters, cls=lambda x, y, z: x, **kwargs)
        kwargs.pop('error_map', None)
        kwargs.pop('content_type', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('FileServer', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        path_format_arguments = {'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', pattern='^[-\\w\\._]+$'), 'workspaceName': self._serialize.url('workspace_name', workspace_name, 'str', max_length=64, min_length=1, pattern='^[-\\w_]+$'), 'fileServerName': self._serialize.url('file_server_name', file_server_name, 'str', max_length=64, min_length=1, pattern='^[-\\w_]+$'), 'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str')}
        if polling is True:
            polling_method = ARMPolling(lro_delay, path_format_arguments=path_format_arguments, **kwargs)
        elif polling is False:
            polling_method = NoPolling()
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        else:
            return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BatchAI/workspaces/{workspaceName}/fileServers/{fileServerName}'}

    def list_by_workspace(self, resource_group_name, workspace_name, file_servers_list_by_workspace_options=None, **kwargs):
        if False:
            while True:
                i = 10
        'Gets a list of File Servers associated with the specified workspace.\n\n        :param resource_group_name: Name of the resource group to which the resource belongs.\n        :type resource_group_name: str\n        :param workspace_name: The name of the workspace. Workspace names can only contain a\n         combination of alphanumeric characters along with dash (-) and underscore (_). The name must be\n         from 1 through 64 characters long.\n        :type workspace_name: str\n        :param file_servers_list_by_workspace_options: Parameter group.\n        :type file_servers_list_by_workspace_options: ~batch_ai.models.FileServersListByWorkspaceOptions\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either FileServerListResult or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~batch_ai.models.FileServerListResult]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        _max_results = None
        if file_servers_list_by_workspace_options is not None:
            _max_results = file_servers_list_by_workspace_options.max_results
        api_version = '2018-05-01'
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                return 10
            header_parameters = {}
            header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
            if not next_link:
                url = self.list_by_workspace.metadata['url']
                path_format_arguments = {'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', pattern='^[-\\w\\._]+$'), 'workspaceName': self._serialize.url('workspace_name', workspace_name, 'str', max_length=64, min_length=1, pattern='^[-\\w_]+$'), 'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                if _max_results is not None:
                    query_parameters['maxresults'] = self._serialize.query('max_results', _max_results, 'int', maximum=1000, minimum=1)
                query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
                request = self._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters = {}
                request = self._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('FileServerListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_workspace.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BatchAI/workspaces/{workspaceName}/fileServers'}