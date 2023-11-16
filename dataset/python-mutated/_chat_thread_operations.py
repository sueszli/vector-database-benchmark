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

class ChatThreadOperations(object):
    """ChatThreadOperations operations.

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
            for i in range(10):
                print('nop')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    def list_chat_read_receipts(self, chat_thread_id, max_page_size=None, skip=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Gets chat message read receipts for a thread.\n\n        Gets chat message read receipts for a thread.\n\n        :param chat_thread_id: Thread id to get the chat message read receipts for.\n        :type chat_thread_id: str\n        :param max_page_size: The maximum number of chat message read receipts to be returned per page.\n        :type max_page_size: int\n        :param skip: Skips chat message read receipts up to a specified position in response.\n        :type skip: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ChatMessageReadReceiptsCollection or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.models.ChatMessageReadReceiptsCollection]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
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
                url = self.list_chat_read_receipts.metadata['url']
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                if max_page_size is not None:
                    query_parameters['maxPageSize'] = self._serialize.query('max_page_size', max_page_size, 'int')
                if skip is not None:
                    query_parameters['skip'] = self._serialize.query('skip', skip, 'int')
                query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
                request = self._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters = {}
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                request = self._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('ChatMessageReadReceiptsCollection', pipeline_response)
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
                raise HttpResponseError(response=response)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_chat_read_receipts.metadata = {'url': '/chat/threads/{chatThreadId}/readReceipts'}

    def send_chat_read_receipt(self, chat_thread_id, send_read_receipt_request, **kwargs):
        if False:
            print('Hello World!')
        'Sends a read receipt event to a thread, on behalf of a user.\n\n        Sends a read receipt event to a thread, on behalf of a user.\n\n        :param chat_thread_id: Thread id to send the read receipt event to.\n        :type chat_thread_id: str\n        :param send_read_receipt_request: Read receipt details.\n        :type send_read_receipt_request: ~azure.communication.chat.models.SendReadReceiptRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.send_chat_read_receipt.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(send_read_receipt_request, 'SendReadReceiptRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        if cls:
            return cls(pipeline_response, None, {})
    send_chat_read_receipt.metadata = {'url': '/chat/threads/{chatThreadId}/readReceipts'}

    def send_chat_message(self, chat_thread_id, send_chat_message_request, **kwargs):
        if False:
            while True:
                i = 10
        'Sends a message to a thread.\n\n        Sends a message to a thread.\n\n        :param chat_thread_id: The thread id to send the message to.\n        :type chat_thread_id: str\n        :param send_chat_message_request: Details of the message to send.\n        :type send_chat_message_request: ~azure.communication.chat.models.SendChatMessageRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SendChatMessageResult, or the result of cls(response)\n        :rtype: ~azure.communication.chat.models.SendChatMessageResult\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.send_chat_message.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(send_chat_message_request, 'SendChatMessageRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        deserialized = self._deserialize('SendChatMessageResult', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    send_chat_message.metadata = {'url': '/chat/threads/{chatThreadId}/messages'}

    def list_chat_messages(self, chat_thread_id, max_page_size=None, start_time=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Gets a list of messages from a thread.\n\n        Gets a list of messages from a thread.\n\n        :param chat_thread_id: The thread id of the message.\n        :type chat_thread_id: str\n        :param max_page_size: The maximum number of messages to be returned per page.\n        :type max_page_size: int\n        :param start_time: The earliest point in time to get messages up to. The timestamp should be in\n         RFC3339 format: ``yyyy-MM-ddTHH:mm:ssZ``.\n        :type start_time: ~datetime.datetime\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ChatMessagesCollection or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.models.ChatMessagesCollection]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                return 10
            header_parameters = {}
            header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
            if not next_link:
                url = self.list_chat_messages.metadata['url']
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
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
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                request = self._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('ChatMessagesCollection', pipeline_response)
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
    list_chat_messages.metadata = {'url': '/chat/threads/{chatThreadId}/messages'}

    def get_chat_message(self, chat_thread_id, chat_message_id, **kwargs):
        if False:
            print('Hello World!')
        'Gets a message by id.\n\n        Gets a message by id.\n\n        :param chat_thread_id: The thread id to which the message was sent.\n        :type chat_thread_id: str\n        :param chat_message_id: The message id.\n        :type chat_message_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ChatMessage, or the result of cls(response)\n        :rtype: ~azure.communication.chat.models.ChatMessage\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        accept = 'application/json'
        url = self.get_chat_message.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str'), 'chatMessageId': self._serialize.url('chat_message_id', chat_message_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.get(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        deserialized = self._deserialize('ChatMessage', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_chat_message.metadata = {'url': '/chat/threads/{chatThreadId}/messages/{chatMessageId}'}

    def update_chat_message(self, chat_thread_id, chat_message_id, update_chat_message_request, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Updates a message.\n\n        Updates a message.\n\n        :param chat_thread_id: The thread id to which the message was sent.\n        :type chat_thread_id: str\n        :param chat_message_id: The message id.\n        :type chat_message_id: str\n        :param update_chat_message_request: Details of the request to update the message.\n        :type update_chat_message_request: ~azure.communication.chat.models.UpdateChatMessageRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/merge-patch+json')
        accept = 'application/json'
        url = self.update_chat_message.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str'), 'chatMessageId': self._serialize.url('chat_message_id', chat_message_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(update_chat_message_request, 'UpdateChatMessageRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.patch(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        if cls:
            return cls(pipeline_response, None, {})
    update_chat_message.metadata = {'url': '/chat/threads/{chatThreadId}/messages/{chatMessageId}'}

    def delete_chat_message(self, chat_thread_id, chat_message_id, **kwargs):
        if False:
            i = 10
            return i + 15
        'Deletes a message.\n\n        Deletes a message.\n\n        :param chat_thread_id: The thread id to which the message was sent.\n        :type chat_thread_id: str\n        :param chat_message_id: The message id.\n        :type chat_message_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        accept = 'application/json'
        url = self.delete_chat_message.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str'), 'chatMessageId': self._serialize.url('chat_message_id', chat_message_id, 'str')}
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
    delete_chat_message.metadata = {'url': '/chat/threads/{chatThreadId}/messages/{chatMessageId}'}

    def list_chat_participants(self, chat_thread_id, max_page_size=None, skip=None, **kwargs):
        if False:
            return 10
        'Gets the participants of a thread.\n\n        Gets the participants of a thread.\n\n        :param chat_thread_id: Thread id to get participants for.\n        :type chat_thread_id: str\n        :param max_page_size: The maximum number of participants to be returned per page.\n        :type max_page_size: int\n        :param skip: Skips participants up to a specified position in response.\n        :type skip: int\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ChatParticipantsCollection or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.models.ChatParticipantsCollection]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            header_parameters = {}
            header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
            if not next_link:
                url = self.list_chat_participants.metadata['url']
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                if max_page_size is not None:
                    query_parameters['maxPageSize'] = self._serialize.query('max_page_size', max_page_size, 'int')
                if skip is not None:
                    query_parameters['skip'] = self._serialize.query('skip', skip, 'int')
                query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
                request = self._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters = {}
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                request = self._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('ChatParticipantsCollection', pipeline_response)
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
                raise HttpResponseError(response=response)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_chat_participants.metadata = {'url': '/chat/threads/{chatThreadId}/participants'}

    def remove_chat_participant(self, chat_thread_id, participant_communication_identifier, **kwargs):
        if False:
            while True:
                i = 10
        'Remove a participant from a thread.\n\n        Remove a participant from a thread.\n\n        :param chat_thread_id: Thread id to remove the participant from.\n        :type chat_thread_id: str\n        :param participant_communication_identifier: Id of the thread participant to remove from the\n         thread.\n        :type participant_communication_identifier: ~azure.communication.chat.models.CommunicationIdentifierModel\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.remove_chat_participant.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(participant_communication_identifier, 'CommunicationIdentifierModel')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        if cls:
            return cls(pipeline_response, None, {})
    remove_chat_participant.metadata = {'url': '/chat/threads/{chatThreadId}/participants/:remove'}

    def add_chat_participants(self, chat_thread_id, add_chat_participants_request, **kwargs):
        if False:
            print('Hello World!')
        'Adds thread participants to a thread. If participants already exist, no change occurs.\n\n        Adds thread participants to a thread. If participants already exist, no change occurs.\n\n        :param chat_thread_id: Id of the thread to add participants to.\n        :type chat_thread_id: str\n        :param add_chat_participants_request: Thread participants to be added to the thread.\n        :type add_chat_participants_request: ~azure.communication.chat.models.AddChatParticipantsRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AddChatParticipantsResult, or the result of cls(response)\n        :rtype: ~azure.communication.chat.models.AddChatParticipantsResult\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.add_chat_participants.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(add_chat_participants_request, 'AddChatParticipantsRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        deserialized = self._deserialize('AddChatParticipantsResult', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    add_chat_participants.metadata = {'url': '/chat/threads/{chatThreadId}/participants/:add'}

    def update_chat_thread_properties(self, chat_thread_id, update_chat_thread_request, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Updates a thread's properties.\n\n        Updates a thread's properties.\n\n        :param chat_thread_id: The id of the thread to update.\n        :type chat_thread_id: str\n        :param update_chat_thread_request: Request payload for updating a chat thread.\n        :type update_chat_thread_request: ~azure.communication.chat.models.UpdateChatThreadRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        "
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/merge-patch+json')
        accept = 'application/json'
        url = self.update_chat_thread_properties.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(update_chat_thread_request, 'UpdateChatThreadRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.patch(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        if cls:
            return cls(pipeline_response, None, {})
    update_chat_thread_properties.metadata = {'url': '/chat/threads/{chatThreadId}'}

    def get_chat_thread_properties(self, chat_thread_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Gets a chat thread's properties.\n\n        Gets a chat thread's properties.\n\n        :param chat_thread_id: Id of the thread.\n        :type chat_thread_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ChatThreadProperties, or the result of cls(response)\n        :rtype: ~azure.communication.chat.models.ChatThreadProperties\n        :raises: ~azure.core.exceptions.HttpResponseError\n        "
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        accept = 'application/json'
        url = self.get_chat_thread_properties.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.get(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        deserialized = self._deserialize('ChatThreadProperties', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_chat_thread_properties.metadata = {'url': '/chat/threads/{chatThreadId}'}

    def send_typing_notification(self, chat_thread_id, send_typing_notification_request=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Posts a typing event to a thread, on behalf of a user.\n\n        Posts a typing event to a thread, on behalf of a user.\n\n        :param chat_thread_id: Id of the thread.\n        :type chat_thread_id: str\n        :param send_typing_notification_request: Details of the typing notification request.\n        :type send_typing_notification_request: ~azure.communication.chat.models.SendTypingNotificationRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {404: ResourceNotFoundError, 409: ResourceExistsError, 401: lambda response: ClientAuthenticationError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 403: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 429: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response)), 503: lambda response: HttpResponseError(response=response, model=self._deserialize(_models.CommunicationErrorResponse, response))}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-09-07'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.send_typing_notification.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True), 'chatThreadId': self._serialize.url('chat_thread_id', chat_thread_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        if send_typing_notification_request is not None:
            body_content = self._serialize.body(send_typing_notification_request, 'SendTypingNotificationRequest')
        else:
            body_content = None
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        if cls:
            return cls(pipeline_response, None, {})
    send_typing_notification.metadata = {'url': '/chat/threads/{chatThreadId}/typing'}