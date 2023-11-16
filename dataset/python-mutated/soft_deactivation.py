import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, TypedDict, Union
from django.conf import settings
from django.db import transaction
from django.db.models import Exists, Max, OuterRef, QuerySet
from django.utils.timezone import now as timezone_now
from sentry_sdk import capture_exception
from zerver.lib.logging_util import log_to_file
from zerver.lib.queue import queue_json_publish
from zerver.lib.utils import assert_is_not_none
from zerver.models import Message, NotificationTriggers, Realm, RealmAuditLog, Recipient, Subscription, UserActivity, UserMessage, UserProfile
logger = logging.getLogger('zulip.soft_deactivation')
log_to_file(logger, settings.SOFT_DEACTIVATION_LOG_PATH)
BULK_CREATE_BATCH_SIZE = 10000

class MissingMessageDict(TypedDict):
    id: int
    recipient__type_id: int

def filter_by_subscription_history(user_profile: UserProfile, all_stream_messages: DefaultDict[int, List[MissingMessageDict]], all_stream_subscription_logs: DefaultDict[int, List[RealmAuditLog]]) -> List[UserMessage]:
    if False:
        print('Hello World!')
    user_messages_to_insert: List[UserMessage] = []
    seen_message_ids: Set[int] = set()

    def store_user_message_to_insert(message: MissingMessageDict) -> None:
        if False:
            return 10
        if message['id'] not in seen_message_ids:
            user_message = UserMessage(user_profile=user_profile, message_id=message['id'], flags=0)
            user_messages_to_insert.append(user_message)
            seen_message_ids.add(message['id'])
    for (stream_id, stream_messages_raw) in all_stream_messages.items():
        stream_subscription_logs = all_stream_subscription_logs[stream_id]
        stream_messages = list(stream_messages_raw)
        for log_entry in stream_subscription_logs:
            if len(stream_messages) == 0:
                break
            event_last_message_id = assert_is_not_none(log_entry.event_last_message_id)
            if log_entry.event_type == RealmAuditLog.SUBSCRIPTION_DEACTIVATED:
                for stream_message in stream_messages:
                    if stream_message['id'] <= event_last_message_id:
                        store_user_message_to_insert(stream_message)
                    else:
                        break
            elif log_entry.event_type in (RealmAuditLog.SUBSCRIPTION_ACTIVATED, RealmAuditLog.SUBSCRIPTION_CREATED):
                initial_msg_count = len(stream_messages)
                for (i, stream_message) in enumerate(stream_messages):
                    if stream_message['id'] > event_last_message_id:
                        stream_messages = stream_messages[i:]
                        break
                final_msg_count = len(stream_messages)
                if initial_msg_count == final_msg_count and stream_messages[-1]['id'] <= event_last_message_id:
                    stream_messages = []
            else:
                raise AssertionError(f'{log_entry.event_type} is not a subscription event.')
        if len(stream_messages) > 0 and stream_subscription_logs[-1].event_type in (RealmAuditLog.SUBSCRIPTION_ACTIVATED, RealmAuditLog.SUBSCRIPTION_CREATED):
            for stream_message in stream_messages:
                store_user_message_to_insert(stream_message)
    return user_messages_to_insert

def add_missing_messages(user_profile: UserProfile) -> None:
    if False:
        while True:
            i = 10
    "This function takes a soft-deactivated user, and computes and adds\n    to the database any UserMessage rows that were not created while\n    the user was soft-deactivated.  The end result is that from the\n    perspective of the message database, it should be impossible to\n    tell that the user was soft-deactivated at all.\n\n    At a high level, the algorithm is as follows:\n\n    * Find all the streams that the user was at any time a subscriber\n      of when or after they were soft-deactivated (`recipient_ids`\n      below).\n\n    * Find all the messages sent to those streams since the user was\n      soft-deactivated.  This will be a superset of the target\n      UserMessages we need to create in two ways: (1) some UserMessage\n      rows will have already been created in do_send_messages because\n      the user had a nonzero set of flags (the fact that we do so in\n      do_send_messages simplifies things considerably, since it means\n      we don't need to inspect message content to look for things like\n      mentions here), and (2) the user might not have been subscribed\n      to all of the streams in recipient_ids for the entire time\n      window.\n\n    * Correct the list from the previous state by excluding those with\n      existing UserMessage rows.\n\n    * Correct the list from the previous state by excluding those\n      where the user wasn't subscribed at the time, using the\n      RealmAuditLog data to determine exactly when the user was\n      subscribed/unsubscribed.\n\n    * Create the UserMessage rows.\n\n    For further documentation, see:\n\n      https://zulip.readthedocs.io/en/latest/subsystems/sending-messages.html#soft-deactivation\n\n    "
    assert user_profile.last_active_message_id is not None
    all_stream_subs = list(Subscription.objects.filter(user_profile=user_profile, recipient__type=Recipient.STREAM).values('recipient_id', 'recipient__type_id'))
    stream_ids = [sub['recipient__type_id'] for sub in all_stream_subs]
    events = [RealmAuditLog.SUBSCRIPTION_CREATED, RealmAuditLog.SUBSCRIPTION_DEACTIVATED, RealmAuditLog.SUBSCRIPTION_ACTIVATED]
    subscription_logs = list(RealmAuditLog.objects.filter(modified_user=user_profile, modified_stream_id__in=stream_ids, event_type__in=events).order_by('event_last_message_id', 'id').only('id', 'event_type', 'modified_stream_id', 'event_last_message_id'))
    all_stream_subscription_logs: DefaultDict[int, List[RealmAuditLog]] = defaultdict(list)
    for log in subscription_logs:
        all_stream_subscription_logs[assert_is_not_none(log.modified_stream_id)].append(log)
    recipient_ids = []
    for sub in all_stream_subs:
        stream_subscription_logs = all_stream_subscription_logs[sub['recipient__type_id']]
        if stream_subscription_logs[-1].event_type == RealmAuditLog.SUBSCRIPTION_DEACTIVATED:
            assert stream_subscription_logs[-1].event_last_message_id is not None
            if stream_subscription_logs[-1].event_last_message_id <= user_profile.last_active_message_id:
                continue
        recipient_ids.append(sub['recipient_id'])
    new_stream_msgs = Message.objects.annotate(has_user_message=Exists(UserMessage.objects.filter(user_profile_id=user_profile, message_id=OuterRef('id')))).filter(has_user_message=0, realm_id=user_profile.realm_id, recipient_id__in=recipient_ids, id__gt=user_profile.last_active_message_id).order_by('id').values('id', 'recipient__type_id')
    stream_messages: DefaultDict[int, List[MissingMessageDict]] = defaultdict(list)
    for msg in new_stream_msgs:
        stream_messages[msg['recipient__type_id']].append(MissingMessageDict(id=msg['id'], recipient__type_id=msg['recipient__type_id']))
    user_messages_to_insert = filter_by_subscription_history(user_profile, stream_messages, all_stream_subscription_logs)
    while len(user_messages_to_insert) > 0:
        (messages, user_messages_to_insert) = (user_messages_to_insert[0:BULK_CREATE_BATCH_SIZE], user_messages_to_insert[BULK_CREATE_BATCH_SIZE:])
        UserMessage.objects.bulk_create(messages)
        user_profile.last_active_message_id = messages[-1].message_id
        user_profile.save(update_fields=['last_active_message_id'])

def do_soft_deactivate_user(user_profile: UserProfile) -> None:
    if False:
        print('Hello World!')
    try:
        user_profile.last_active_message_id = UserMessage.objects.filter(user_profile=user_profile).order_by('-message_id')[0].message_id
    except IndexError:
        last_message = Message.objects.last()
        assert last_message is not None
        user_profile.last_active_message_id = last_message.id
    user_profile.long_term_idle = True
    user_profile.save(update_fields=['long_term_idle', 'last_active_message_id'])
    logger.info('Soft deactivated user %s', user_profile.id)

def do_soft_deactivate_users(users: Union[Sequence[UserProfile], QuerySet[UserProfile]]) -> List[UserProfile]:
    if False:
        print('Hello World!')
    BATCH_SIZE = 100
    users_soft_deactivated = []
    while True:
        (user_batch, users) = (users[0:BATCH_SIZE], users[BATCH_SIZE:])
        if len(user_batch) == 0:
            break
        with transaction.atomic():
            realm_logs = []
            for user in user_batch:
                do_soft_deactivate_user(user)
                event_time = timezone_now()
                log = RealmAuditLog(realm=user.realm, modified_user=user, event_type=RealmAuditLog.USER_SOFT_DEACTIVATED, event_time=event_time)
                realm_logs.append(log)
                users_soft_deactivated.append(user)
            RealmAuditLog.objects.bulk_create(realm_logs)
        logger.info('Soft-deactivated batch of %s users; %s remain to process', len(user_batch), len(users))
    return users_soft_deactivated

def do_auto_soft_deactivate_users(inactive_for_days: int, realm: Optional[Realm]) -> List[UserProfile]:
    if False:
        i = 10
        return i + 15
    filter_kwargs: Dict[str, Realm] = {}
    if realm is not None:
        filter_kwargs = dict(user_profile__realm=realm)
    users_to_deactivate = get_users_for_soft_deactivation(inactive_for_days, filter_kwargs)
    users_deactivated = do_soft_deactivate_users(users_to_deactivate)
    if not settings.AUTO_CATCH_UP_SOFT_DEACTIVATED_USERS:
        logger.info('Not catching up users since AUTO_CATCH_UP_SOFT_DEACTIVATED_USERS is off')
        return users_deactivated
    if realm is not None:
        filter_kwargs = dict(realm=realm)
    users_to_catch_up = get_soft_deactivated_users_for_catch_up(filter_kwargs)
    do_catch_up_soft_deactivated_users(users_to_catch_up)
    return users_deactivated

def reactivate_user_if_soft_deactivated(user_profile: UserProfile) -> Union[UserProfile, None]:
    if False:
        for i in range(10):
            print('nop')
    if user_profile.long_term_idle:
        add_missing_messages(user_profile)
        user_profile.long_term_idle = False
        user_profile.save(update_fields=['long_term_idle'])
        RealmAuditLog.objects.create(realm=user_profile.realm, modified_user=user_profile, event_type=RealmAuditLog.USER_SOFT_ACTIVATED, event_time=timezone_now())
        logger.info('Soft reactivated user %s', user_profile.id)
        return user_profile
    return None

def get_users_for_soft_deactivation(inactive_for_days: int, filter_kwargs: Any) -> List[UserProfile]:
    if False:
        print('Hello World!')
    users_activity = list(UserActivity.objects.filter(user_profile__is_active=True, user_profile__is_bot=False, user_profile__long_term_idle=False, **filter_kwargs).values('user_profile_id').annotate(last_visit=Max('last_visit')))
    today = timezone_now()
    user_ids_to_deactivate = [user_activity['user_profile_id'] for user_activity in users_activity if (today - user_activity['last_visit']).days > inactive_for_days]
    users_to_deactivate = list(UserProfile.objects.filter(id__in=user_ids_to_deactivate))
    return users_to_deactivate

def do_soft_activate_users(users: List[UserProfile]) -> List[UserProfile]:
    if False:
        i = 10
        return i + 15
    return [user_activated for user_profile in users if (user_activated := reactivate_user_if_soft_deactivated(user_profile)) is not None]

def do_catch_up_soft_deactivated_users(users: Iterable[UserProfile]) -> List[UserProfile]:
    if False:
        while True:
            i = 10
    users_caught_up = []
    failures = []
    for user_profile in users:
        if user_profile.long_term_idle:
            try:
                add_missing_messages(user_profile)
                users_caught_up.append(user_profile)
            except Exception:
                capture_exception()
                failures.append(user_profile)
    logger.info('Caught up %d soft-deactivated users', len(users_caught_up))
    if failures:
        logger.error('Failed to catch up %d soft-deactivated users', len(failures))
    return users_caught_up

def get_soft_deactivated_users_for_catch_up(filter_kwargs: Any) -> QuerySet[UserProfile]:
    if False:
        print('Hello World!')
    users_to_catch_up = UserProfile.objects.filter(long_term_idle=True, is_active=True, is_bot=False, **filter_kwargs)
    return users_to_catch_up

def queue_soft_reactivation(user_profile_id: int) -> None:
    if False:
        return 10
    event = {'type': 'soft_reactivate', 'user_profile_id': user_profile_id}
    queue_json_publish('deferred_work', event)

def soft_reactivate_if_personal_notification(user_profile: UserProfile, unique_triggers: Set[str], mentioned_user_group_name: Optional[str]) -> None:
    if False:
        return 10
    "When we're about to send an email/push notification to a\n    long_term_idle user, it's very likely that the user will try to\n    return to Zulip. As a result, it makes sense to optimistically\n    soft-reactivate that user, to give them a good return experience.\n\n    It's important that we do nothing for stream wildcard or group mentions,\n    because soft-reactivating an entire realm would be very expensive\n    (and we can't easily check the group's size). The caller is\n    responsible for passing a mentioned_user_group_name that is None\n    for messages that contain both a personal mention and a group\n    mention.\n    "
    if not user_profile.long_term_idle:
        return
    direct_message = NotificationTriggers.DIRECT_MESSAGE in unique_triggers
    personal_mention = NotificationTriggers.MENTION in unique_triggers and mentioned_user_group_name is None
    topic_wildcard_mention = any((trigger in unique_triggers for trigger in [NotificationTriggers.TOPIC_WILDCARD_MENTION, NotificationTriggers.TOPIC_WILDCARD_MENTION_IN_FOLLOWED_TOPIC]))
    if not direct_message and (not personal_mention) and (not topic_wildcard_mention):
        return
    queue_soft_reactivation(user_profile.id)