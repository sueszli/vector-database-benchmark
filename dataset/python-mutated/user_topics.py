import datetime
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
from django.db import transaction
from django.db.models import QuerySet
from django.utils.timezone import now as timezone_now
from sqlalchemy.sql import ClauseElement, and_, column, not_, or_
from sqlalchemy.types import Integer
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.topic import topic_match_sa
from zerver.lib.types import UserTopicDict
from zerver.models import UserProfile, UserTopic, get_stream

def get_user_topics(user_profile: UserProfile, include_deactivated: bool=False, include_stream_name: bool=False, visibility_policy: Optional[int]=None) -> List[UserTopicDict]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Fetches UserTopic objects associated with the target user.\n    * include_deactivated: Whether to include those associated with\n      deactivated streams.\n    * include_stream_name: Whether to include stream names in the\n      returned dictionaries.\n    * visibility_policy: If specified, returns only UserTopic objects\n      with the specified visibility_policy value.\n    '
    query = UserTopic.objects.filter(user_profile=user_profile)
    if visibility_policy is not None:
        query = query.filter(visibility_policy=visibility_policy)
    if not include_deactivated:
        query = query.filter(stream__deactivated=False)
    rows = query.values('stream_id', 'stream__name', 'topic_name', 'last_updated', 'visibility_policy')
    result = []
    for row in rows:
        user_topic_dict: UserTopicDict = {'stream_id': row['stream_id'], 'topic_name': row['topic_name'], 'visibility_policy': row['visibility_policy'], 'last_updated': datetime_to_timestamp(row['last_updated'])}
        if include_stream_name:
            user_topic_dict['stream__name'] = row['stream__name']
        result.append(user_topic_dict)
    return result

def get_topic_mutes(user_profile: UserProfile, include_deactivated: bool=False) -> List[Tuple[str, str, int]]:
    if False:
        while True:
            i = 10
    user_topics = get_user_topics(user_profile=user_profile, include_deactivated=include_deactivated, include_stream_name=True, visibility_policy=UserTopic.VisibilityPolicy.MUTED)
    return [(user_topic['stream__name'], user_topic['topic_name'], user_topic['last_updated']) for user_topic in user_topics]

def set_topic_visibility_policy(user_profile: UserProfile, topics: List[List[str]], visibility_policy: int, last_updated: Optional[datetime.datetime]=None) -> None:
    if False:
        print('Hello World!')
    '\n    This is only used in tests.\n    '
    UserTopic.objects.filter(user_profile=user_profile, visibility_policy=visibility_policy).delete()
    if last_updated is None:
        last_updated = timezone_now()
    for (stream_name, topic_name) in topics:
        stream = get_stream(stream_name, user_profile.realm)
        recipient_id = stream.recipient_id
        assert recipient_id is not None
        bulk_set_user_topic_visibility_policy_in_database(user_profiles=[user_profile], stream_id=stream.id, recipient_id=recipient_id, topic_name=topic_name, visibility_policy=visibility_policy, last_updated=last_updated)

def get_topic_visibility_policy(user_profile: UserProfile, stream_id: int, topic_name: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    try:
        user_topic = UserTopic.objects.get(user_profile=user_profile, stream_id=stream_id, topic_name__iexact=topic_name)
        visibility_policy = user_topic.visibility_policy
    except UserTopic.DoesNotExist:
        visibility_policy = UserTopic.VisibilityPolicy.INHERIT
    return visibility_policy

@transaction.atomic(savepoint=False)
def bulk_set_user_topic_visibility_policy_in_database(user_profiles: List[UserProfile], stream_id: int, topic_name: str, *, visibility_policy: int, recipient_id: Optional[int]=None, last_updated: Optional[datetime.datetime]=None) -> List[UserProfile]:
    if False:
        print('Hello World!')
    rows = UserTopic.objects.filter(user_profile__in=user_profiles, stream_id=stream_id, topic_name__iexact=topic_name).select_related('user_profile', 'user_profile__realm')
    user_profiles_with_visibility_policy = [row.user_profile for row in rows]
    user_profiles_without_visibility_policy = list(set(user_profiles) - set(user_profiles_with_visibility_policy))
    if visibility_policy == UserTopic.VisibilityPolicy.INHERIT:
        for user_profile in user_profiles_without_visibility_policy:
            logging.info("User %s tried to remove visibility_policy, which actually doesn't exist", user_profile.id)
        rows.delete()
        return user_profiles_with_visibility_policy
    assert last_updated is not None
    assert recipient_id is not None
    user_profiles_seeking_visibility_policy_update: List[UserProfile] = []
    for row in rows:
        duplicate_request: bool = row.visibility_policy == visibility_policy
        if duplicate_request:
            logging.info('User %s tried to set visibility_policy to its current value of %s', row.user_profile_id, visibility_policy)
            continue
        user_profiles_seeking_visibility_policy_update.append(row.user_profile)
    if user_profiles_seeking_visibility_policy_update:
        rows.filter(user_profile__in=user_profiles_seeking_visibility_policy_update).update(visibility_policy=visibility_policy, last_updated=last_updated)
    if user_profiles_without_visibility_policy:
        UserTopic.objects.bulk_create((UserTopic(user_profile=user_profile, stream_id=stream_id, recipient_id=recipient_id, topic_name=topic_name, last_updated=last_updated, visibility_policy=visibility_policy) for user_profile in user_profiles_without_visibility_policy))
    return user_profiles_seeking_visibility_policy_update + user_profiles_without_visibility_policy

def topic_has_visibility_policy(user_profile: UserProfile, stream_id: int, topic_name: str, visibility_policy: int) -> bool:
    if False:
        i = 10
        return i + 15
    if visibility_policy == UserTopic.VisibilityPolicy.INHERIT:
        has_user_topic_row = UserTopic.objects.filter(user_profile=user_profile, stream_id=stream_id, topic_name__iexact=topic_name).exists()
        return not has_user_topic_row
    has_visibility_policy = UserTopic.objects.filter(user_profile=user_profile, stream_id=stream_id, topic_name__iexact=topic_name, visibility_policy=visibility_policy).exists()
    return has_visibility_policy

def exclude_topic_mutes(conditions: List[ClauseElement], user_profile: UserProfile, stream_id: Optional[int]) -> List[ClauseElement]:
    if False:
        i = 10
        return i + 15
    query = UserTopic.objects.filter(user_profile=user_profile, visibility_policy=UserTopic.VisibilityPolicy.MUTED)
    if stream_id is not None:
        query = query.filter(stream_id=stream_id)
    rows = query.values('recipient_id', 'topic_name')
    if not rows:
        return conditions

    class RecipientTopicDict(TypedDict):
        recipient_id: int
        topic_name: str

    def mute_cond(row: RecipientTopicDict) -> ClauseElement:
        if False:
            i = 10
            return i + 15
        recipient_id = row['recipient_id']
        topic_name = row['topic_name']
        stream_cond = column('recipient_id', Integer) == recipient_id
        topic_cond = topic_match_sa(topic_name)
        return and_(stream_cond, topic_cond)
    condition = not_(or_(*map(mute_cond, rows)))
    return [*conditions, condition]

def build_get_topic_visibility_policy(user_profile: UserProfile) -> Callable[[int, str], int]:
    if False:
        return 10
    "Prefetch the visibility policies the user has configured for\n    various topics.\n\n    The prefetching helps to avoid the db queries later in the loop\n    to determine the user's visibility policy for a topic.\n    "
    rows = UserTopic.objects.filter(user_profile=user_profile).values('recipient_id', 'topic_name', 'visibility_policy')
    topic_to_visibility_policy: Dict[Tuple[int, str], int] = defaultdict(int)
    for row in rows:
        recipient_id = row['recipient_id']
        topic_name = row['topic_name']
        visibility_policy = row['visibility_policy']
        topic_to_visibility_policy[recipient_id, topic_name] = visibility_policy

    def get_topic_visibility_policy(recipient_id: int, topic: str) -> int:
        if False:
            while True:
                i = 10
        return topic_to_visibility_policy[recipient_id, topic.lower()]
    return get_topic_visibility_policy

def get_users_with_user_topic_visibility_policy(stream_id: int, topic_name: str) -> QuerySet[UserTopic]:
    if False:
        while True:
            i = 10
    return UserTopic.objects.filter(stream_id=stream_id, topic_name__iexact=topic_name).select_related('user_profile', 'user_profile__realm')