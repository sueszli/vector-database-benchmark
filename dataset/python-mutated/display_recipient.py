from typing import Dict, List, Optional, Set, Tuple, TypedDict
from django_stubs_ext import ValuesQuerySet
from zerver.lib.cache import bulk_cached_fetch, cache_with_key, display_recipient_cache_key, generic_bulk_cached_fetch, single_user_display_recipient_cache_key
from zerver.lib.types import DisplayRecipientT, UserDisplayRecipient
from zerver.models import Recipient, Stream, UserProfile, bulk_get_huddle_user_ids
display_recipient_fields = ['id', 'email', 'full_name', 'is_mirror_dummy']

class TinyStreamResult(TypedDict):
    recipient_id: int
    name: str

def get_display_recipient_cache_key(recipient_id: int, recipient_type: int, recipient_type_id: Optional[int]) -> str:
    if False:
        print('Hello World!')
    return display_recipient_cache_key(recipient_id)

@cache_with_key(get_display_recipient_cache_key, timeout=3600 * 24 * 7)
def get_display_recipient_remote_cache(recipient_id: int, recipient_type: int, recipient_type_id: Optional[int]) -> List[UserDisplayRecipient]:
    if False:
        return 10
    '\n    This returns an appropriate object describing the recipient of a\n    direct message (whether individual or group).\n\n    It will be an array of dicts for each recipient.\n\n    Do not use this for streams.\n    '
    assert recipient_type != Recipient.STREAM
    user_profile_list = UserProfile.objects.filter(subscription__recipient_id=recipient_id).order_by('id').values(*display_recipient_fields)
    return list(user_profile_list)

def user_dict_id_fetcher(user_dict: UserDisplayRecipient) -> int:
    if False:
        i = 10
        return i + 15
    return user_dict['id']

def bulk_fetch_single_user_display_recipients(uids: List[int]) -> Dict[int, UserDisplayRecipient]:
    if False:
        print('Hello World!')
    return bulk_cached_fetch(cache_key_function=single_user_display_recipient_cache_key, query_function=lambda ids: list(UserProfile.objects.filter(id__in=ids).values(*display_recipient_fields)), object_ids=uids, id_fetcher=user_dict_id_fetcher)

def bulk_fetch_stream_names(recipient_tuples: Set[Tuple[int, int, int]]) -> Dict[int, str]:
    if False:
        print('Hello World!')
    '\n    Takes set of tuples of the form (recipient_id, recipient_type, recipient_type_id)\n    Returns dict mapping recipient_id to corresponding display_recipient\n    '
    if len(recipient_tuples) == 0:
        return {}
    recipient_id_to_stream_id = {tup[0]: tup[2] for tup in recipient_tuples}
    recipient_ids = [tup[0] for tup in recipient_tuples]

    def get_tiny_stream_rows(recipient_ids: List[int]) -> ValuesQuerySet[Stream, TinyStreamResult]:
        if False:
            while True:
                i = 10
        stream_ids = [recipient_id_to_stream_id[recipient_id] for recipient_id in recipient_ids]
        return Stream.objects.filter(id__in=stream_ids).values('recipient_id', 'name')

    def get_recipient_id(row: TinyStreamResult) -> int:
        if False:
            return 10
        return row['recipient_id']

    def get_name(row: TinyStreamResult) -> str:
        if False:
            i = 10
            return i + 15
        return row['name']
    stream_display_recipients: Dict[int, str] = generic_bulk_cached_fetch(cache_key_function=display_recipient_cache_key, query_function=get_tiny_stream_rows, object_ids=recipient_ids, id_fetcher=get_recipient_id, cache_transformer=get_name, setter=lambda obj: obj, extractor=lambda obj: obj)
    return stream_display_recipients

def bulk_fetch_user_display_recipients(recipient_tuples: Set[Tuple[int, int, int]]) -> Dict[int, List[UserDisplayRecipient]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes set of tuples of the form (recipient_id, recipient_type, recipient_type_id)\n    Returns dict mapping recipient_id to corresponding display_recipient\n    '
    if len(recipient_tuples) == 0:
        return {}
    get_recipient_id = lambda tup: tup[0]
    get_type = lambda tup: tup[1]
    personal_tuples = [tup for tup in recipient_tuples if get_type(tup) == Recipient.PERSONAL]
    huddle_tuples = [tup for tup in recipient_tuples if get_type(tup) == Recipient.HUDDLE]
    huddle_recipient_ids = [get_recipient_id(tup) for tup in huddle_tuples]
    huddle_recipient_id_to_user_ids = bulk_get_huddle_user_ids(huddle_recipient_ids)
    user_ids_to_fetch: Set[int] = set()
    for (ignore_recipient_id, ignore_recipient_type, user_id) in personal_tuples:
        user_ids_to_fetch.add(user_id)
    for recipient_id in huddle_recipient_ids:
        huddle_user_ids = huddle_recipient_id_to_user_ids[recipient_id]
        user_ids_to_fetch |= huddle_user_ids
    user_display_recipients = bulk_fetch_single_user_display_recipients(list(user_ids_to_fetch))
    result = {}
    for (recipient_id, ignore_recipient_type, user_id) in personal_tuples:
        display_recipients = [user_display_recipients[user_id]]
        result[recipient_id] = display_recipients
    for recipient_id in huddle_recipient_ids:
        user_ids = sorted(huddle_recipient_id_to_user_ids[recipient_id])
        display_recipients = [user_display_recipients[user_id] for user_id in user_ids]
        result[recipient_id] = display_recipients
    return result

def bulk_fetch_display_recipients(recipient_tuples: Set[Tuple[int, int, int]]) -> Dict[int, DisplayRecipientT]:
    if False:
        while True:
            i = 10
    '\n    Takes set of tuples of the form (recipient_id, recipient_type, recipient_type_id)\n    Returns dict mapping recipient_id to corresponding display_recipient\n    '
    stream_recipients = {recipient for recipient in recipient_tuples if recipient[1] == Recipient.STREAM}
    personal_and_huddle_recipients = recipient_tuples - stream_recipients
    stream_display_recipients = bulk_fetch_stream_names(stream_recipients)
    personal_and_huddle_display_recipients = bulk_fetch_user_display_recipients(personal_and_huddle_recipients)
    return {**stream_display_recipients, **personal_and_huddle_display_recipients}