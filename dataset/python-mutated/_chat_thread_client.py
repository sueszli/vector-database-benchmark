from typing import TYPE_CHECKING
from urllib.parse import urlparse
from azure.core.tracing.decorator import distributed_trace
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from ._shared.user_credential import CommunicationTokenCredential
from ._shared.models import CommunicationIdentifier
from ._generated import AzureCommunicationChatService
from ._generated.models import AddChatParticipantsRequest, SendReadReceiptRequest, SendChatMessageRequest, SendTypingNotificationRequest, UpdateChatMessageRequest, UpdateChatThreadRequest, ChatMessageType, SendChatMessageResult
from ._models import ChatParticipant, ChatMessage, ChatMessageReadReceipt, ChatThreadProperties
from ._communication_identifier_serializer import serialize_identifier
from ._utils import CommunicationErrorResponseConverter
from ._version import SDK_MONIKER
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union, Tuple
    from datetime import datetime
    from azure.core.paging import ItemPaged

class ChatThreadClient(object):
    """A client to interact with the AzureCommunicationService Chat gateway.
    Instances of this class is normally retrieved by ChatClient.get_chat_thread_client()

    This client provides operations to add participant(s) to chat thread, remove participant from
    chat thread, send message, delete message, update message, send typing notifications,
    send and list read receipt

    :ivar thread_id: Chat thread id.
    :vartype thread_id: str

    :param str endpoint:
        The endpoint of the Azure Communication resource.
    :param CommunicationTokenCredential credential:
        The credentials with which to authenticate. The value contains a User
        Access Token
    :param str thread_id:
        The unique thread id.

    .. admonition:: Example:

        .. literalinclude:: ../samples/chat_thread_client_sample.py
            :start-after: [START create_chat_thread_client]
            :end-before: [END create_chat_thread_client]
            :language: python
            :dedent: 8
            :caption: Creating the ChatThreadClient.
    """

    def __init__(self, endpoint, credential, thread_id, **kwargs):
        if False:
            i = 10
            return i + 15
        if not thread_id:
            raise ValueError('thread_id can not be None or empty')
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
        self._thread_id = thread_id
        self._endpoint = endpoint
        self._credential = credential
        self._client = AzureCommunicationChatService(endpoint, authentication_policy=BearerTokenCredentialPolicy(self._credential), sdk_moniker=SDK_MONIKER, **kwargs)

    @property
    def thread_id(self):
        if False:
            return 10
        '\n        Gets the thread id from the client.\n\n        :rtype: str\n        '
        return self._thread_id

    @distributed_trace
    def get_properties(self, **kwargs):
        if False:
            while True:
                i = 10
        'Gets the properties of the chat thread.\n\n        :return: ChatThreadProperties\n        :rtype: ~azure.communication.chat.ChatThreadProperties\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START get_thread]\n                :end-before: [END get_thread]\n                :language: python\n                :dedent: 8\n                :caption: Retrieving chat thread properties by chat thread id.\n        '
        chat_thread = self._client.chat_thread.get_chat_thread_properties(self._thread_id, **kwargs)
        return ChatThreadProperties._from_generated(chat_thread)

    @distributed_trace
    def update_topic(self, topic=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Updates a thread's properties.\n\n        :param topic: Thread topic. If topic is not specified, the update will succeed but\n         chat thread properties will not be changed.\n        :type topic: str\n        :return: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START update_topic]\n                :end-before: [END update_topic]\n                :language: python\n                :dedent: 8\n                :caption: Updating chat thread.\n        "
        update_topic_request = UpdateChatThreadRequest(topic=topic)
        return self._client.chat_thread.update_chat_thread_properties(chat_thread_id=self._thread_id, update_chat_thread_request=update_topic_request, **kwargs)

    @distributed_trace
    def send_read_receipt(self, message_id, **kwargs):
        if False:
            i = 10
            return i + 15
        'Posts a read receipt event to a chat thread, on behalf of a user.\n\n        :param message_id: Required. Id of the latest message read by current user.\n        :type message_id: str\n        :return: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START send_read_receipt]\n                :end-before: [END send_read_receipt]\n                :language: python\n                :dedent: 8\n                :caption: Sending read receipt of a chat message.\n        '
        if not message_id:
            raise ValueError('message_id cannot be None.')
        post_read_receipt_request = SendReadReceiptRequest(chat_message_id=message_id)
        return self._client.chat_thread.send_chat_read_receipt(self._thread_id, send_read_receipt_request=post_read_receipt_request, **kwargs)

    @distributed_trace
    def list_read_receipts(self, **kwargs):
        if False:
            while True:
                i = 10
        'Gets read receipts for a thread.\n\n        :keyword int results_per_page: The maximum number of chat message read receipts to be returned per page.\n        :keyword int skip: Skips chat message read receipts up to a specified position in response.\n        :return: An iterator like instance of ChatMessageReadReceipt\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.ChatMessageReadReceipt]\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START list_read_receipts]\n                :end-before: [END list_read_receipts]\n                :language: python\n                :dedent: 8\n                :caption: Listing read receipts.\n        '
        results_per_page = kwargs.pop('results_per_page', None)
        skip = kwargs.pop('skip', None)
        return self._client.chat_thread.list_chat_read_receipts(self._thread_id, max_page_size=results_per_page, skip=skip, cls=lambda objs: [ChatMessageReadReceipt._from_generated(x) for x in objs], **kwargs)

    @distributed_trace
    def send_typing_notification(self, **kwargs):
        if False:
            print('Hello World!')
        'Posts a typing event to a thread, on behalf of a user.\n\n        :keyword str sender_display_name: The display name of the typing notification sender. This property\n         is used to populate sender name for push notifications.\n        :return: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START send_typing_notification]\n                :end-before: [END send_typing_notification]\n                :language: python\n                :dedent: 8\n                :caption: Send typing notification.\n        '
        sender_display_name = kwargs.pop('sender_display_name', None)
        send_typing_notification_request = SendTypingNotificationRequest(sender_display_name=sender_display_name)
        return self._client.chat_thread.send_typing_notification(chat_thread_id=self._thread_id, send_typing_notification_request=send_typing_notification_request, **kwargs)

    @distributed_trace
    def send_message(self, content, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Sends a message to a thread.\n\n        :param content: Required. Chat message content.\n        :type content: str\n        :keyword chat_message_type:\n            The chat message type. Possible values include: "text", "html". Default: ChatMessageType.TEXT\n        :paramtype chat_message_type: Union[str, ~azure.communication.chat.ChatMessageType]\n        :keyword str sender_display_name: The display name of the message sender. This property is used to\n            populate sender name for push notifications.\n        :keyword dict[str, str] metadata: Message metadata.\n        :return: SendChatMessageResult\n        :rtype: ~azure.communication.chat.SendChatMessageResult\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START send_message]\n                :end-before: [END send_message]\n                :language: python\n                :dedent: 8\n                :caption: Sending a message.\n        '
        if not content:
            raise ValueError('content cannot be None.')
        chat_message_type = kwargs.pop('chat_message_type', None)
        if chat_message_type is None:
            chat_message_type = ChatMessageType.TEXT
        elif not isinstance(chat_message_type, ChatMessageType):
            try:
                chat_message_type = ChatMessageType.__getattr__(chat_message_type)
            except Exception:
                raise ValueError('chat_message_type: {message_type} is not acceptable'.format(message_type=chat_message_type))
        if chat_message_type not in [ChatMessageType.TEXT, ChatMessageType.HTML]:
            raise ValueError("chat_message_type: {message_type} can be only 'text' or 'html'".format(message_type=chat_message_type))
        sender_display_name = kwargs.pop('sender_display_name', None)
        metadata = kwargs.pop('metadata', None)
        create_message_request = SendChatMessageRequest(content=content, type=chat_message_type, sender_display_name=sender_display_name, metadata=metadata)
        send_chat_message_result = self._client.chat_thread.send_chat_message(chat_thread_id=self._thread_id, send_chat_message_request=create_message_request, **kwargs)
        return send_chat_message_result

    @distributed_trace
    def get_message(self, message_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Gets a message by id.\n\n        :param message_id: Required. The message id.\n        :type message_id: str\n        :return: ChatMessage\n        :rtype: ~azure.communication.chat.ChatMessage\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START get_message]\n                :end-before: [END get_message]\n                :language: python\n                :dedent: 8\n                :caption: Retrieving a message by message id.\n        '
        if not message_id:
            raise ValueError('message_id cannot be None.')
        chat_message = self._client.chat_thread.get_chat_message(self._thread_id, message_id, **kwargs)
        return ChatMessage._from_generated(chat_message)

    @distributed_trace
    def list_messages(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Gets a list of messages from a thread.\n\n        :keyword int results_per_page: The maximum number of messages to be returned per page.\n        :keyword ~datetime.datetime start_time: The earliest point in time to get messages up to.\n        The timestamp should be in RFC3339 format: ``yyyy-MM-ddTHH:mm:ssZ``.\n        :return: An iterator like instance of ChatMessage\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.ChatMessage]\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START list_messages]\n                :end-before: [END list_messages]\n                :language: python\n                :dedent: 8\n                :caption: Listing messages of a chat thread.\n        '
        results_per_page = kwargs.pop('results_per_page', None)
        start_time = kwargs.pop('start_time', None)
        a = self._client.chat_thread.list_chat_messages(self._thread_id, max_page_size=results_per_page, start_time=start_time, cls=lambda objs: [ChatMessage._from_generated(x) for x in objs], **kwargs)
        return a

    @distributed_trace
    def update_message(self, message_id, content=None, **kwargs):
        if False:
            while True:
                i = 10
        'Updates a message.\n\n        :param message_id: Required. The message id.\n        :type message_id: str\n        :param content: Chat message content.\n        :type content: str\n        :keyword dict[str, str] metadata: Message metadata.\n        :return: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START update_message]\n                :end-before: [END update_message]\n                :language: python\n                :dedent: 8\n                :caption: Updating an already sent message.\n        '
        if not message_id:
            raise ValueError('message_id cannot be None.')
        metadata = kwargs.pop('metadata', None)
        update_message_request = UpdateChatMessageRequest(content=content, metadata=metadata)
        return self._client.chat_thread.update_chat_message(chat_thread_id=self._thread_id, chat_message_id=message_id, update_chat_message_request=update_message_request, **kwargs)

    @distributed_trace
    def delete_message(self, message_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Deletes a message.\n\n        :param message_id: Required. The message id.\n        :type message_id: str\n        :return: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START delete_message]\n                :end-before: [END delete_message]\n                :language: python\n                :dedent: 8\n                :caption: Deleting a message.\n        '
        if not message_id:
            raise ValueError('message_id cannot be None.')
        return self._client.chat_thread.delete_chat_message(chat_thread_id=self._thread_id, chat_message_id=message_id, **kwargs)

    @distributed_trace
    def list_participants(self, **kwargs):
        if False:
            while True:
                i = 10
        'Gets the participants of a thread.\n\n        :keyword int results_per_page: The maximum number of participants to be returned per page.\n        :keyword int skip: Skips participants up to a specified position in response.\n        :return: An iterator like instance of ChatParticipant\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.chat.ChatParticipant]\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START list_participants]\n                :end-before: [END list_participants]\n                :language: python\n                :dedent: 8\n                :caption: Listing participants of chat thread.\n        '
        results_per_page = kwargs.pop('results_per_page', None)
        skip = kwargs.pop('skip', None)
        return self._client.chat_thread.list_chat_participants(self._thread_id, max_page_size=results_per_page, skip=skip, cls=lambda objs: [ChatParticipant._from_generated(x) for x in objs], **kwargs)

    @distributed_trace
    def add_participants(self, thread_participants, **kwargs):
        if False:
            print('Hello World!')
        'Adds thread participants to a thread. If participants already exist, no change occurs.\n\n        If all participants are added successfully, then an empty list is returned;\n        otherwise, a list of tuple(chat_thread_participant, chat_error) is returned,\n        of failed participants and its respective error\n\n        :param thread_participants: Thread participants to be added to the thread.\n        :type thread_participants: List[~azure.communication.chat.ChatParticipant]\n        :return: List[Tuple[ChatParticipant, ChatError]]\n        :rtype: List[Tuple[~azure.communication.chat.ChatParticipant, ~azure.communication.chat.ChatError]]\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START add_participants]\n                :end-before: [END add_participants]\n                :language: python\n                :dedent: 8\n                :caption: Adding participants to chat thread.\n        '
        response = []
        if thread_participants:
            participants = [m._to_generated() for m in thread_participants]
            add_thread_participants_request = AddChatParticipantsRequest(participants=participants)
            add_chat_participants_result = self._client.chat_thread.add_chat_participants(chat_thread_id=self._thread_id, add_chat_participants_request=add_thread_participants_request, **kwargs)
            if hasattr(add_chat_participants_result, 'invalid_participants') and add_chat_participants_result.invalid_participants is not None:
                response = CommunicationErrorResponseConverter._convert(participants=thread_participants, chat_errors=add_chat_participants_result.invalid_participants)
        return response

    @distributed_trace
    def remove_participant(self, identifier, **kwargs):
        if False:
            return 10
        'Remove a participant from a thread.\n\n        :param identifier: Required. Identifier of the thread participant to remove from the thread.\n        :type identifier: ~azure.communication.chat.CommunicationIdentifier\n        :return: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/chat_thread_client_sample.py\n                :start-after: [START remove_participant]\n                :end-before: [END remove_participant]\n                :language: python\n                :dedent: 8\n                :caption: Removing participant from chat thread.\n        '
        if not identifier:
            raise ValueError('identifier cannot be None.')
        return self._client.chat_thread.remove_chat_participant(chat_thread_id=self._thread_id, participant_communication_identifier=serialize_identifier(identifier), **kwargs)

    def close(self):
        if False:
            print('Hello World!')
        return self._client.close()

    def __enter__(self):
        if False:
            print('Hello World!')
        self._client.__enter__()
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        self._client.__exit__(*args)