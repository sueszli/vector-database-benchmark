from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import orjson
from django.db import connection
from django.db.models import Q, QuerySet, Subquery
from sqlalchemy.sql import ColumnElement, column, func, literal
from sqlalchemy.types import Boolean, Text
from zerver.lib.request import REQ
from zerver.lib.types import EditHistoryEvent
from zerver.models import Message, Reaction, Stream, UserMessage, UserProfile
ORIG_TOPIC = 'orig_subject'
TOPIC_NAME = 'subject'
TOPIC_LINKS = 'topic_links'
MATCH_TOPIC = 'match_subject'
RESOLVED_TOPIC_PREFIX = '✔ '
EXPORT_TOPIC_NAME = 'subject'
'\nThe following functions are for user-facing APIs\nwhere we\'ll want to support "subject" for a while.\n'

def get_topic_from_message_info(message_info: Dict[str, Any]) -> str:
    if False:
        while True:
            i = 10
    "\n    Use this where you are getting dicts that are based off of messages\n    that may come from the outside world, especially from third party\n    APIs and bots.\n\n    We prefer 'topic' to 'subject' here.  We expect at least one field\n    to be present (or the caller must know how to handle KeyError).\n    "
    if 'topic' in message_info:
        return message_info['topic']
    return message_info['subject']

def REQ_topic() -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    return REQ(whence='topic', aliases=['subject'], converter=lambda var_name, x: x.strip(), default=None)
'\nTRY TO KEEP THIS DIVIDING LINE.\n\nBelow this line we want to make it so that functions are only\nusing "subject" in the DB sense, and nothing customer facing.\n\n'
DB_TOPIC_NAME = 'subject'
MESSAGE__TOPIC = 'message__subject'

def topic_match_sa(topic_name: str) -> ColumnElement[Boolean]:
    if False:
        for i in range(10):
            print('nop')
    topic_cond = func.upper(column('subject', Text)) == func.upper(literal(topic_name))
    return topic_cond

def get_resolved_topic_condition_sa() -> ColumnElement[Boolean]:
    if False:
        for i in range(10):
            print('nop')
    resolved_topic_cond = column('subject', Text).startswith(RESOLVED_TOPIC_PREFIX)
    return resolved_topic_cond

def topic_column_sa() -> ColumnElement[Text]:
    if False:
        i = 10
        return i + 15
    return column('subject', Text)

def filter_by_topic_name_via_message(query: QuerySet[UserMessage], topic_name: str) -> QuerySet[UserMessage]:
    if False:
        for i in range(10):
            print('nop')
    return query.filter(message__subject__iexact=topic_name)

def messages_for_topic(realm_id: int, stream_recipient_id: int, topic_name: str) -> QuerySet[Message]:
    if False:
        print('Hello World!')
    return Message.objects.filter(realm_id=realm_id, recipient_id=stream_recipient_id, subject__iexact=topic_name)

def save_message_for_edit_use_case(message: Message) -> None:
    if False:
        while True:
            i = 10
    message.save(update_fields=[TOPIC_NAME, 'content', 'rendered_content', 'rendered_content_version', 'last_edit_time', 'edit_history', 'has_attachment', 'has_image', 'has_link', 'recipient_id'])

def user_message_exists_for_topic(user_profile: UserProfile, recipient_id: int, topic_name: str) -> bool:
    if False:
        i = 10
        return i + 15
    return UserMessage.objects.filter(user_profile=user_profile, message__recipient_id=recipient_id, message__subject__iexact=topic_name).exists()

def update_edit_history(message: Message, last_edit_time: datetime, edit_history_event: EditHistoryEvent) -> None:
    if False:
        i = 10
        return i + 15
    message.last_edit_time = last_edit_time
    if message.edit_history is not None:
        edit_history: List[EditHistoryEvent] = orjson.loads(message.edit_history)
        edit_history.insert(0, edit_history_event)
    else:
        edit_history = [edit_history_event]
    message.edit_history = orjson.dumps(edit_history).decode()

def update_messages_for_topic_edit(acting_user: UserProfile, edited_message: Message, propagate_mode: str, orig_topic_name: str, topic_name: Optional[str], new_stream: Optional[Stream], old_stream: Stream, edit_history_event: EditHistoryEvent, last_edit_time: datetime) -> List[Message]:
    if False:
        print('Hello World!')
    propagate_query = Q(recipient_id=old_stream.recipient_id, subject__iexact=orig_topic_name)
    if propagate_mode == 'change_all':
        propagate_query = propagate_query & ~Q(id=edited_message.id)
    if propagate_mode == 'change_later':
        propagate_query = propagate_query & Q(id__gt=edited_message.id)
    messages = Message.objects.filter(propagate_query, realm_id=old_stream.realm_id).select_related(*Message.DEFAULT_SELECT_RELATED)
    update_fields = ['edit_history', 'last_edit_time']
    if new_stream is not None:
        from zerver.lib.message import bulk_access_messages
        messages_list = bulk_access_messages(acting_user, messages, stream=old_stream)
    else:
        messages_list = list(messages)
    if new_stream is not None:
        update_fields.append('recipient')
        for m in messages_list:
            assert new_stream.recipient is not None
            m.recipient = new_stream.recipient
    if topic_name is not None:
        update_fields.append('subject')
        for m in messages_list:
            m.set_topic_name(topic_name)
    for message in messages_list:
        update_edit_history(message, last_edit_time, edit_history_event)
    Message.objects.bulk_update(messages_list, update_fields, batch_size=100)
    return messages_list

def generate_topic_history_from_db_rows(rows: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    canonical_topic_names: Dict[str, Tuple[int, str]] = {}
    rows = sorted(rows, key=lambda tup: tup[1])
    for (topic_name, max_message_id) in rows:
        canonical_name = topic_name.lower()
        canonical_topic_names[canonical_name] = (max_message_id, topic_name)
    history = []
    for (max_message_id, topic_name) in canonical_topic_names.values():
        history.append(dict(name=topic_name, max_id=max_message_id))
    return sorted(history, key=lambda x: -x['max_id'])

def get_topic_history_for_public_stream(realm_id: int, recipient_id: int) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    cursor = connection.cursor()
    query = '\n    SELECT\n        "zerver_message"."subject" as topic,\n        max("zerver_message".id) as max_message_id\n    FROM "zerver_message"\n    WHERE (\n        "zerver_message"."realm_id" = %s AND\n        "zerver_message"."recipient_id" = %s\n    )\n    GROUP BY (\n        "zerver_message"."subject"\n    )\n    ORDER BY max("zerver_message".id) DESC\n    '
    cursor.execute(query, [realm_id, recipient_id])
    rows = cursor.fetchall()
    cursor.close()
    return generate_topic_history_from_db_rows(rows)

def get_topic_history_for_stream(user_profile: UserProfile, recipient_id: int, public_history: bool) -> List[Dict[str, Any]]:
    if False:
        return 10
    if public_history:
        return get_topic_history_for_public_stream(user_profile.realm_id, recipient_id)
    cursor = connection.cursor()
    query = '\n    SELECT\n        "zerver_message"."subject" as topic,\n        max("zerver_message".id) as max_message_id\n    FROM "zerver_message"\n    INNER JOIN "zerver_usermessage" ON (\n        "zerver_usermessage"."message_id" = "zerver_message"."id"\n    )\n    WHERE (\n        "zerver_usermessage"."user_profile_id" = %s AND\n        "zerver_message"."realm_id" = %s AND\n        "zerver_message"."recipient_id" = %s\n    )\n    GROUP BY (\n        "zerver_message"."subject"\n    )\n    ORDER BY max("zerver_message".id) DESC\n    '
    cursor.execute(query, [user_profile.id, user_profile.realm_id, recipient_id])
    rows = cursor.fetchall()
    cursor.close()
    return generate_topic_history_from_db_rows(rows)

def get_topic_resolution_and_bare_name(stored_name: str) -> Tuple[bool, str]:
    if False:
        i = 10
        return i + 15
    '\n    Resolved topics are denoted only by a title change, not by a boolean toggle in a database column. This\n    method inspects the topic name and returns a tuple of:\n\n    - Whether the topic has been resolved\n    - The topic name with the resolution prefix, if present in stored_name, removed\n    '
    if stored_name.startswith(RESOLVED_TOPIC_PREFIX):
        return (True, stored_name[len(RESOLVED_TOPIC_PREFIX):])
    return (False, stored_name)

def participants_for_topic(realm_id: int, recipient_id: int, topic_name: str) -> Set[int]:
    if False:
        print('Hello World!')
    '\n    Users who either sent or reacted to the messages in the topic.\n    The function is expensive for large numbers of messages in the topic.\n    '
    messages = Message.objects.filter(realm_id=realm_id, recipient_id=recipient_id, subject__iexact=topic_name)
    participants = set(UserProfile.objects.filter(Q(id__in=Subquery(messages.values('sender_id'))) | Q(id__in=Subquery(Reaction.objects.filter(message__in=messages).values('user_profile_id')))).values_list('id', flat=True))
    return participants