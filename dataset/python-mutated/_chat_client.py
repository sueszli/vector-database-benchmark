from typing import TYPE_CHECKING
from uuid import uuid4
from urllib.parse import urlparse
from azure.core.tracing.decorator import distributed_trace
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from ._chat_thread_client import ChatThreadClient
from ._shared.user_credential import CommunicationTokenCredential
from ._generated import AzureCommunicationChatService
from ._generated.models import CreateChatThreadRequest
from ._models import ChatThreadProperties, CreateChatThreadResult
from ._utils import _to_utc_datetime, return_response, CommunicationErrorResponseConverter
from ._version import SDK_MONIKER
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
    from datetime import datetime
    from azure.core.paging import ItemPaged

class ChatClient(object):
    """A client to interact with the AzureCommunicationService Chat gateway.

    This client provides operations to create chat thread, delete chat thread,
    get chat thread client by thread id, list chat threads.

    :param str endpoint:
        The endpoint of the Azure Communication resource.
    :param CommunicationTokenCredential credential:
        The credentials with which to authenticate.

    .. admonition:: Example:

        .. literalinclude:: ../samples/chat_client_sample.py
            :start-after: [START create_chat_client]
            :end-before: [END create_chat_client]
            :language: python
            :dedent: 8
            :caption: Creating the ChatClient from a URL and token.
    """

    def __init__(self, endpoint, credential, **kwargs):
        if False:
            return 10
        if not credential:
            raise ValueError('credential can not be None')
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError:
            raise ValueError('Host URL must be a string')
        parsed_url = urlparse(endpoint.rstrip('/'))
        if not parsed_url.netloc:
            raise ValueError('Invalid URL: {}'.format(endpoint))
        self._endpoint = endpoint
        self._credential = credential
        self._client = AzureCommunicationChatService(self._endpoint, authentication_policy=BearerTokenCredentialPolicy(self._credential), sdk_moniker=SDK_MONIKER, **kwargs)

    @distributed_trace
    def get_chat_thread_client(self, thread_id, **kwargs):
        if False:
            return 10
        '\n        Get ChatThreadClient by providing a thread_id.\n\n        :param thread_id: Required. The thread id.\n        :type thread_id: str\n        :return: ChatThreadClient\n        :rtype: ~azure.communication.chat.ChatThreadClient\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_client_sample.py\n                :start-after: [START get_chat_thread_client]\n                :end-before: [END get_chat_thread_client]\n                :language: python\n                :dedent: 8\n                :caption: Retrieving the ChatThreadClient from an existing chat thread id.\n        '
        if not thread_id:
            raise ValueError('thread_id cannot be None.')
        return ChatThreadClient(endpoint=self._endpoint, credential=self._credential, thread_id=thread_id, **kwargs)

    @distributed_trace
    def create_chat_thread(self, topic, **kwargs):
        if False:
            i = 10
            return i + 15
        'Creates a chat thread.\n\n        :param topic: Required. The thread topic.\n        :type topic: str\n        :keyword thread_participants: Optional. Participants to be added to the thread.\n        :paramtype thread_participants: List[~azure.communication.chat.ChatParticipant]\n        :keyword idempotency_token: Optional. If specified, the client directs that the request is\n         repeatable; that is, the client can make the request multiple times with the same\n         Idempotency_Token and get back an appropriate response without the server executing the\n         request multiple times. The value of the Idempotency_Token is an opaque string\n         representing a client-generated, globally unique for all time, identifier for the request. If not\n         specified, a new unique id would be generated.\n        :paramtype idempotency_token: str\n        :return: CreateChatThreadResult\n        :rtype: ~azure.communication.chat.CreateChatThreadResult\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_client_sample.py\n                :start-after: [START create_thread]\n                :end-before: [END create_thread]\n                :language: python\n                :dedent: 8\n                :caption: Creating a new chat thread.\n        '
        if not topic:
            raise ValueError('topic cannot be None.')
        idempotency_token = kwargs.pop('idempotency_token', None)
        if idempotency_token is None:
            idempotency_token = str(uuid4())
        thread_participants = kwargs.pop('thread_participants', None)
        participants = []
        if thread_participants is not None:
            participants = [m._to_generated() for m in thread_participants]
        create_thread_request = CreateChatThreadRequest(topic=topic, participants=participants)
        create_chat_thread_result = self._client.chat.create_chat_thread(create_chat_thread_request=create_thread_request, repeatability_request_id=idempotency_token, **kwargs)
        errors = None
        if hasattr(create_chat_thread_result, 'errors') and create_chat_thread_result.errors is not None:
            errors = CommunicationErrorResponseConverter._convert(participants=[thread_participants], chat_errors=create_chat_thread_result.invalid_participants)
        chat_thread_properties = ChatThreadProperties._from_generated(create_chat_thread_result.chat_thread)
        create_chat_thread_result = CreateChatThreadResult(chat_thread=chat_thread_properties, errors=errors)
        return create_chat_thread_result

    @distributed_trace
    def list_chat_threads(self, **kwargs):
        if False:
            while True:
                i = 10
        'Gets the list of chat threads of a user.\n\n        :keyword int results_per_page: The maximum number of chat threads returned per page.\n        :keyword ~datetime.datetime start_time: The earliest point in time to get chat threads up to.\n        :return: An iterator like instance of ChatThreadItem\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.ChatThreadItem]\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_client_sample.py\n                :start-after: [START list_threads]\n                :end-before: [END list_threads]\n                :language: python\n                :dedent: 8\n                :caption: Listing chat threads.\n        '
        results_per_page = kwargs.pop('results_per_page', None)
        start_time = kwargs.pop('start_time', None)
        return self._client.chat.list_chat_threads(max_page_size=results_per_page, start_time=start_time, **kwargs)

    @distributed_trace
    def delete_chat_thread(self, thread_id, **kwargs):
        if False:
            i = 10
            return i + 15
        'Deletes a chat thread.\n\n        :param thread_id: Required. Thread id to delete.\n        :type thread_id: str\n        :return: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_client_sample.py\n                :start-after: [START delete_thread]\n                :end-before: [END delete_thread]\n                :language: python\n                :dedent: 8\n                :caption: Deleting a chat thread.\n        '
        if not thread_id:
            raise ValueError('thread_id cannot be None.')
        return self._client.chat.delete_chat_thread(thread_id, **kwargs)

    def close(self):
        if False:
            i = 10
            return i + 15
        self._client.close()

    def __enter__(self):
        if False:
            while True:
                i = 10
        self._client.__enter__()
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self._client.__exit__(*args)