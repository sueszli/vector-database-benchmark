from typing import Iterable, List, TypedDict
from zerver.lib import retention
from zerver.lib.retention import move_messages_to_archive
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id
from zerver.models import Message, Realm, UserMessage, UserProfile
from zerver.tornado.django_api import send_event_on_commit

class DeleteMessagesEvent(TypedDict, total=False):
    type: str
    message_ids: List[int]
    message_type: str
    topic: str
    stream_id: int

def do_delete_messages(realm: Realm, messages: Iterable[Message]) -> None:
    if False:
        for i in range(10):
            print('nop')
    messages = list(messages)
    message_ids = [message.id for message in messages]
    if not message_ids:
        return
    event: DeleteMessagesEvent = {'type': 'delete_message', 'message_ids': message_ids}
    sample_message = messages[0]
    message_type = 'stream'
    users_to_notify = []
    if not sample_message.is_stream_message():
        assert len(messages) == 1
        message_type = 'private'
        ums = UserMessage.objects.filter(message_id__in=message_ids)
        users_to_notify = [um.user_profile_id for um in ums]
        archiving_chunk_size = retention.MESSAGE_BATCH_SIZE
    if message_type == 'stream':
        stream_id = sample_message.recipient.type_id
        event['stream_id'] = stream_id
        event['topic'] = sample_message.topic_name()
        subscriptions = get_active_subscriptions_for_stream_id(stream_id, include_deactivated_users=False)
        subscriptions = subscriptions.exclude(user_profile__long_term_idle=True)
        users_to_notify = list(subscriptions.values_list('user_profile_id', flat=True))
        archiving_chunk_size = retention.STREAM_MESSAGE_BATCH_SIZE
    move_messages_to_archive(message_ids, realm=realm, chunk_size=archiving_chunk_size)
    event['message_type'] = message_type
    send_event_on_commit(realm, event, users_to_notify)

def do_delete_messages_by_sender(user: UserProfile) -> None:
    if False:
        i = 10
        return i + 15
    message_ids = list(Message.objects.filter(realm_id=user.realm_id, sender=user).values_list('id', flat=True).order_by('id'))
    if message_ids:
        move_messages_to_archive(message_ids, chunk_size=retention.STREAM_MESSAGE_BATCH_SIZE)