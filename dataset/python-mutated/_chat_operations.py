import datetime
from typing import TYPE_CHECKING
import warnings
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from .. import models as _models
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

class ChatOperations(object):
    """ChatOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~azure.communication.chat.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    def create_chat_thread(self, create_chat_thread_request, repeatability_request_id=None, **kwargs):
        if False:
            return 10
        'Creates a chat thread.\n\n        Creates a chat thread.\n\n        :param create_chat_thread_request: Request payload for creating a chat thread.\n        :type create_chat_thread_request: ~azure.communication.chat.models.CreateChatThreadRequest\n        :param repeatability_request_id: If specified, the client directs that the request is\n         repeatable; that is, that the client can make the request multiple times with the same\n         Repeatability-Request-Id and get back an appropriate response without the server executing the\n         request multiple times. The value of the Repeatability-Request-Id is an opaque string\n         representing a client-generated, globally unique for all time, identifier for the request. It\n         is recommended to use version 4 (random) UUIDs.\n        :type repeatability_request_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CreateChatThreadResult, or the result of cls(response)\n        :rtype: ~azure.communication.chat.models.CreateChatThreadResult\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.create_chat_thread.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        if repeatability_request_id is not None:
            header_parameters['repeatability-request-id'] = self._serialize.header('repeatability_request_id', repeatability_request_id, 'str')
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(create_chat_thread_request, 'CreateChatThreadRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        deserialized = self._deserialize('CreateChatThreadResult', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    create_chat_thread.metadata = {'url': '/chat/threads'}

    def list_chat_threads(self, max_page_size=None, start_time=None, **kwargs):
        if False:
            return 10
        'Gets the list of chat threads of a user.\n\n        Gets the list of chat threads of a user.\n\n        :param max_page_size: The maximum number of chat threads returned per page.\n        :type max_page_size: int\n        :param start_time: The earliest point in time to get chat threads up to. The timestamp should\n         be in RFC3339 format: ``yyyy-MM-ddTHH:mm:ssZ``.\n        :type start_time: ~datetime.datetime\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ChatThreadsItemCollection or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.models.ChatThreadsItemCollection]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            header_parameters = {}
            header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
            if not next_link:
                url = self.list_chat_threads.metadata['url']
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                if max_page_size is not None:
                    query_parameters['maxPageSize'] = self._serialize.query('max_page_size', max_page_size, 'int')
                if start_time is not None:
                    query_parameters['startTime'] = self._serialize.query('start_time', start_time, 'iso-8601')
                query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
                request = self._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters = {}
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                url = self._client.format_url(url, **path_format_arguments)
                request = self._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('ChatThreadsItemCollection', pipeline_response)
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
                raise HttpResponseError(response=response)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_chat_threads.metadata = {'url': '/chat/threads'}

    def delete_chat_thread(self, chat_thread_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Deletes a thread.\n\n        Deletes a thread.\n\n        :param chat_thread_id: Id of the thread to be deleted.\n        :type chat_thread_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        accept = 'application/json'
        url = self.delete_chat_thread.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.delete(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        if cls:
            return cls(pipeline_response, None, {})
    delete_chat_thread.metadata = {'url': '/chat/threads/{chatThreadId}'}