import itertools
from operator import itemgetter
from typing import Any, Callable, Collection, Dict, Iterable, List, Mapping, Optional, Set, Tuple
from django.core.exceptions import ValidationError
from django.db import connection
from django.db.models import QuerySet
from django.utils.translation import gettext as _
from psycopg2.sql import SQL
from zerver.lib.email_mirror_helpers import encode_email_address_helper
from zerver.lib.exceptions import JsonableError
from zerver.lib.stream_color import STREAM_ASSIGNMENT_COLORS
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id, get_stream_subscriptions_for_user
from zerver.lib.stream_traffic import get_average_weekly_stream_traffic, get_streams_traffic
from zerver.lib.streams import get_web_public_streams_queryset, subscribed_to_stream
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.types import NeverSubscribedStreamDict, RawStreamDict, RawSubscriptionDict, SubscriptionInfo, SubscriptionStreamDict
from zerver.models import Realm, Stream, Subscription, UserProfile, get_active_streams

def get_web_public_subs(realm: Realm) -> SubscriptionInfo:
    if False:
        print('Hello World!')
    color_idx = 0

    def get_next_color() -> str:
        if False:
            for i in range(10):
                print('nop')
        nonlocal color_idx
        color = STREAM_ASSIGNMENT_COLORS[color_idx]
        color_idx = (color_idx + 1) % len(STREAM_ASSIGNMENT_COLORS)
        return color
    subscribed = []
    for stream in get_web_public_streams_queryset(realm):
        can_remove_subscribers_group_id = stream.can_remove_subscribers_group_id
        date_created = datetime_to_timestamp(stream.date_created)
        description = stream.description
        first_message_id = stream.first_message_id
        history_public_to_subscribers = stream.history_public_to_subscribers
        invite_only = stream.invite_only
        is_announcement_only = stream.stream_post_policy == Stream.STREAM_POST_POLICY_ADMINS
        is_web_public = stream.is_web_public
        message_retention_days = stream.message_retention_days
        name = stream.name
        rendered_description = stream.rendered_description
        stream_id = stream.id
        stream_post_policy = stream.stream_post_policy
        audible_notifications = True
        color = get_next_color()
        desktop_notifications = True
        email_address = ''
        email_notifications = True
        in_home_view = True
        is_muted = False
        pin_to_top = False
        push_notifications = True
        stream_weekly_traffic = get_average_weekly_stream_traffic(stream.id, stream.date_created, {})
        wildcard_mentions_notify = True
        sub = SubscriptionStreamDict(audible_notifications=audible_notifications, can_remove_subscribers_group=can_remove_subscribers_group_id, color=color, date_created=date_created, description=description, desktop_notifications=desktop_notifications, email_address=email_address, email_notifications=email_notifications, first_message_id=first_message_id, history_public_to_subscribers=history_public_to_subscribers, in_home_view=in_home_view, invite_only=invite_only, is_announcement_only=is_announcement_only, is_muted=is_muted, is_web_public=is_web_public, message_retention_days=message_retention_days, name=name, pin_to_top=pin_to_top, push_notifications=push_notifications, rendered_description=rendered_description, stream_id=stream_id, stream_post_policy=stream_post_policy, stream_weekly_traffic=stream_weekly_traffic, wildcard_mentions_notify=wildcard_mentions_notify)
        subscribed.append(sub)
    return SubscriptionInfo(subscriptions=subscribed, unsubscribed=[], never_subscribed=[])

def build_stream_dict_for_sub(user: UserProfile, sub_dict: RawSubscriptionDict, raw_stream_dict: RawStreamDict, recent_traffic: Optional[Dict[int, int]]) -> SubscriptionStreamDict:
    if False:
        while True:
            i = 10
    can_remove_subscribers_group_id = raw_stream_dict['can_remove_subscribers_group_id']
    date_created = datetime_to_timestamp(raw_stream_dict['date_created'])
    description = raw_stream_dict['description']
    first_message_id = raw_stream_dict['first_message_id']
    history_public_to_subscribers = raw_stream_dict['history_public_to_subscribers']
    invite_only = raw_stream_dict['invite_only']
    is_web_public = raw_stream_dict['is_web_public']
    message_retention_days = raw_stream_dict['message_retention_days']
    name = raw_stream_dict['name']
    rendered_description = raw_stream_dict['rendered_description']
    stream_id = raw_stream_dict['id']
    stream_post_policy = raw_stream_dict['stream_post_policy']
    color = sub_dict['color']
    is_muted = sub_dict['is_muted']
    pin_to_top = sub_dict['pin_to_top']
    audible_notifications = sub_dict['audible_notifications']
    desktop_notifications = sub_dict['desktop_notifications']
    email_notifications = sub_dict['email_notifications']
    push_notifications = sub_dict['push_notifications']
    wildcard_mentions_notify = sub_dict['wildcard_mentions_notify']
    in_home_view = not is_muted
    is_announcement_only = raw_stream_dict['stream_post_policy'] == Stream.STREAM_POST_POLICY_ADMINS
    if recent_traffic is not None:
        stream_weekly_traffic = get_average_weekly_stream_traffic(raw_stream_dict['id'], raw_stream_dict['date_created'], recent_traffic)
    else:
        stream_weekly_traffic = None
    email_address = encode_email_address_helper(raw_stream_dict['name'], raw_stream_dict['email_token'], show_sender=True)
    return SubscriptionStreamDict(audible_notifications=audible_notifications, can_remove_subscribers_group=can_remove_subscribers_group_id, color=color, date_created=date_created, description=description, desktop_notifications=desktop_notifications, email_address=email_address, email_notifications=email_notifications, first_message_id=first_message_id, history_public_to_subscribers=history_public_to_subscribers, in_home_view=in_home_view, invite_only=invite_only, is_announcement_only=is_announcement_only, is_muted=is_muted, is_web_public=is_web_public, message_retention_days=message_retention_days, name=name, pin_to_top=pin_to_top, push_notifications=push_notifications, rendered_description=rendered_description, stream_id=stream_id, stream_post_policy=stream_post_policy, stream_weekly_traffic=stream_weekly_traffic, wildcard_mentions_notify=wildcard_mentions_notify)

def build_stream_dict_for_never_sub(raw_stream_dict: RawStreamDict, recent_traffic: Optional[Dict[int, int]]) -> NeverSubscribedStreamDict:
    if False:
        print('Hello World!')
    can_remove_subscribers_group_id = raw_stream_dict['can_remove_subscribers_group_id']
    date_created = datetime_to_timestamp(raw_stream_dict['date_created'])
    description = raw_stream_dict['description']
    first_message_id = raw_stream_dict['first_message_id']
    history_public_to_subscribers = raw_stream_dict['history_public_to_subscribers']
    invite_only = raw_stream_dict['invite_only']
    is_web_public = raw_stream_dict['is_web_public']
    message_retention_days = raw_stream_dict['message_retention_days']
    name = raw_stream_dict['name']
    rendered_description = raw_stream_dict['rendered_description']
    stream_id = raw_stream_dict['id']
    stream_post_policy = raw_stream_dict['stream_post_policy']
    if recent_traffic is not None:
        stream_weekly_traffic = get_average_weekly_stream_traffic(raw_stream_dict['id'], raw_stream_dict['date_created'], recent_traffic)
    else:
        stream_weekly_traffic = None
    is_announcement_only = raw_stream_dict['stream_post_policy'] == Stream.STREAM_POST_POLICY_ADMINS
    return NeverSubscribedStreamDict(can_remove_subscribers_group=can_remove_subscribers_group_id, date_created=date_created, description=description, first_message_id=first_message_id, history_public_to_subscribers=history_public_to_subscribers, invite_only=invite_only, is_announcement_only=is_announcement_only, is_web_public=is_web_public, message_retention_days=message_retention_days, name=name, rendered_description=rendered_description, stream_id=stream_id, stream_post_policy=stream_post_policy, stream_weekly_traffic=stream_weekly_traffic)

def validate_user_access_to_subscribers(user_profile: Optional[UserProfile], stream: Stream) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Validates whether the user can view the subscribers of a stream.  Raises a JsonableError if:\n    * The user and the stream are in different realms\n    * The realm is MIT and the stream is not invite only.\n    * The stream is invite only, requesting_user is passed, and that user\n      does not subscribe to the stream.\n    '
    validate_user_access_to_subscribers_helper(user_profile, {'realm_id': stream.realm_id, 'is_web_public': stream.is_web_public, 'invite_only': stream.invite_only}, lambda user_profile: subscribed_to_stream(user_profile, stream.id))

def validate_user_access_to_subscribers_helper(user_profile: Optional[UserProfile], stream_dict: Mapping[str, Any], check_user_subscribed: Callable[[UserProfile], bool]) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Helper for validate_user_access_to_subscribers that doesn't require\n    a full stream object.  This function is a bit hard to read,\n    because it is carefully optimized for performance in the two code\n    paths we call it from:\n\n    * In `bulk_get_subscriber_user_ids`, we already know whether the\n    user was subscribed via `sub_dict`, and so we want to avoid a\n    database query at all (especially since it calls this in a loop);\n    * In `validate_user_access_to_subscribers`, we want to only check\n    if the user is subscribed when we absolutely have to, since it\n    costs a database query.\n\n    The `check_user_subscribed` argument is a function that reports\n    whether the user is subscribed to the stream.\n\n    Note also that we raise a ValidationError in cases where the\n    caller is doing the wrong thing (maybe these should be\n    AssertionErrors), and JsonableError for 400 type errors.\n    "
    if user_profile is None:
        raise ValidationError('Missing user to validate access for')
    if user_profile.realm_id != stream_dict['realm_id']:
        raise ValidationError('Requesting user not in given realm')
    if stream_dict['is_web_public']:
        return
    if user_profile.is_guest and check_user_subscribed(user_profile):
        return
    if not user_profile.can_access_public_streams() and (not stream_dict['invite_only']):
        raise JsonableError(_('Subscriber data is not available for this stream'))
    if user_profile.is_realm_admin:
        return
    if stream_dict['invite_only'] and (not check_user_subscribed(user_profile)):
        raise JsonableError(_('Unable to retrieve subscribers for private stream'))

def bulk_get_subscriber_user_ids(stream_dicts: Collection[Mapping[str, Any]], user_profile: UserProfile, subscribed_stream_ids: Set[int]) -> Dict[int, List[int]]:
    if False:
        return 10
    'sub_dict maps stream_id => whether the user is subscribed to that stream.'
    target_stream_dicts = []
    is_subscribed: bool
    check_user_subscribed = lambda user_profile: is_subscribed
    for stream_dict in stream_dicts:
        stream_id = stream_dict['id']
        is_subscribed = stream_id in subscribed_stream_ids
        try:
            validate_user_access_to_subscribers_helper(user_profile, stream_dict, check_user_subscribed)
        except JsonableError:
            continue
        target_stream_dicts.append(stream_dict)
    recip_to_stream_id = {stream['recipient_id']: stream['id'] for stream in target_stream_dicts}
    recipient_ids = sorted((stream['recipient_id'] for stream in target_stream_dicts))
    result: Dict[int, List[int]] = {stream['id']: [] for stream in stream_dicts}
    if not recipient_ids:
        return result
    '\n    The raw SQL below leads to more than a 2x speedup when tested with\n    20k+ total subscribers.  (For large realms with lots of default\n    streams, this function deals with LOTS of data, so it is important\n    to optimize.)\n    '
    query = SQL('\n        SELECT\n            zerver_subscription.recipient_id,\n            zerver_subscription.user_profile_id\n        FROM\n            zerver_subscription\n        WHERE\n            zerver_subscription.recipient_id in %(recipient_ids)s AND\n            zerver_subscription.active AND\n            zerver_subscription.is_user_active\n        ORDER BY\n            zerver_subscription.recipient_id,\n            zerver_subscription.user_profile_id\n        ')
    cursor = connection.cursor()
    cursor.execute(query, {'recipient_ids': tuple(recipient_ids)})
    rows = cursor.fetchall()
    cursor.close()
    '\n    Using groupby/itemgetter here is important for performance, at scale.\n    It makes it so that all interpreter overhead is just O(N) in nature.\n    '
    for (recip_id, recip_rows) in itertools.groupby(rows, itemgetter(0)):
        user_profile_ids = [r[1] for r in recip_rows]
        stream_id = recip_to_stream_id[recip_id]
        result[stream_id] = list(user_profile_ids)
    return result

def get_subscribers_query(stream: Stream, requesting_user: Optional[UserProfile]) -> QuerySet[Subscription]:
    if False:
        while True:
            i = 10
    "Build a query to get the subscribers list for a stream, raising a JsonableError if:\n\n    'realm' is optional in stream.\n\n    The caller can refine this query with select_related(), values(), etc. depending\n    on whether it wants objects or just certain fields\n    "
    validate_user_access_to_subscribers(requesting_user, stream)
    return get_active_subscriptions_for_stream_id(stream.id, include_deactivated_users=False)

def gather_subscriptions_helper(user_profile: UserProfile, include_subscribers: bool=True) -> SubscriptionInfo:
    if False:
        for i in range(10):
            print('nop')
    realm = user_profile.realm
    all_streams = get_active_streams(realm).values(*Stream.API_FIELDS, 'realm_id', 'recipient_id', 'email_token')
    recip_id_to_stream_id: Dict[int, int] = {stream['recipient_id']: stream['id'] for stream in all_streams}
    all_streams_map: Dict[int, RawStreamDict] = {stream['id']: stream for stream in all_streams}
    sub_dicts_query: Iterable[RawSubscriptionDict] = get_stream_subscriptions_for_user(user_profile).values(*Subscription.API_FIELDS, 'recipient_id', 'active').order_by('recipient_id')
    sub_dicts: List[RawSubscriptionDict] = [sub_dict for sub_dict in sub_dicts_query if recip_id_to_stream_id.get(sub_dict['recipient_id'])]

    def get_stream_id(sub_dict: RawSubscriptionDict) -> int:
        if False:
            return 10
        return recip_id_to_stream_id[sub_dict['recipient_id']]
    traffic_stream_ids = {get_stream_id(sub_dict) for sub_dict in sub_dicts}
    recent_traffic = get_streams_traffic(stream_ids=traffic_stream_ids, realm=realm)
    subscribed: List[SubscriptionStreamDict] = []
    unsubscribed: List[SubscriptionStreamDict] = []
    never_subscribed: List[NeverSubscribedStreamDict] = []
    sub_unsub_stream_ids = set()
    for sub_dict in sub_dicts:
        stream_id = get_stream_id(sub_dict)
        sub_unsub_stream_ids.add(stream_id)
        raw_stream_dict = all_streams_map[stream_id]
        stream_dict = build_stream_dict_for_sub(user=user_profile, sub_dict=sub_dict, raw_stream_dict=raw_stream_dict, recent_traffic=recent_traffic)
        is_active = sub_dict['active']
        if is_active:
            subscribed.append(stream_dict)
        else:
            unsubscribed.append(stream_dict)
    if user_profile.can_access_public_streams():
        never_subscribed_stream_ids = set(all_streams_map) - sub_unsub_stream_ids
    else:
        web_public_stream_ids = {stream['id'] for stream in all_streams if stream['is_web_public']}
        never_subscribed_stream_ids = web_public_stream_ids - sub_unsub_stream_ids
    never_subscribed_streams = [all_streams_map[stream_id] for stream_id in never_subscribed_stream_ids]
    for raw_stream_dict in never_subscribed_streams:
        is_public = not raw_stream_dict['invite_only']
        if is_public or user_profile.is_realm_admin:
            slim_stream_dict = build_stream_dict_for_never_sub(raw_stream_dict=raw_stream_dict, recent_traffic=recent_traffic)
            never_subscribed.append(slim_stream_dict)
    if include_subscribers:
        subscribed_stream_ids = {get_stream_id(sub_dict) for sub_dict in sub_dicts if sub_dict['active']}
        subscriber_map = bulk_get_subscriber_user_ids(all_streams, user_profile, subscribed_stream_ids)
        for lst in [subscribed, unsubscribed]:
            for stream_dict in lst:
                assert isinstance(stream_dict['stream_id'], int)
                stream_id = stream_dict['stream_id']
                stream_dict['subscribers'] = subscriber_map[stream_id]
        for slim_stream_dict in never_subscribed:
            assert isinstance(slim_stream_dict['stream_id'], int)
            stream_id = slim_stream_dict['stream_id']
            slim_stream_dict['subscribers'] = subscriber_map[stream_id]
    subscribed.sort(key=lambda x: x['name'])
    unsubscribed.sort(key=lambda x: x['name'])
    never_subscribed.sort(key=lambda x: x['name'])
    return SubscriptionInfo(subscriptions=subscribed, unsubscribed=unsubscribed, never_subscribed=never_subscribed)

def gather_subscriptions(user_profile: UserProfile, include_subscribers: bool=False) -> Tuple[List[SubscriptionStreamDict], List[SubscriptionStreamDict]]:
    if False:
        return 10
    helper_result = gather_subscriptions_helper(user_profile, include_subscribers=include_subscribers)
    subscribed = helper_result.subscriptions
    unsubscribed = helper_result.unsubscribed
    return (subscribed, unsubscribed)