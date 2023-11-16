import datetime
import logging
from typing import Any, Callable, Dict, Iterable, Tuple
from django.conf import settings
from django.contrib.sessions.models import Session
from django.db import connection
from django.db.models import QuerySet
from django.utils.timezone import now as timezone_now
from django_stubs_ext import ValuesQuerySet
from analytics.models import RealmCount
from zerver.lib.cache import cache_set_many, get_remote_cache_requests, get_remote_cache_time, user_profile_by_api_key_cache_key, user_profile_cache_key
from zerver.lib.safe_session_cached_db import SessionStore
from zerver.lib.sessions import session_engine
from zerver.lib.users import get_all_api_keys
from zerver.models import Client, UserProfile, get_client_cache_key

def user_cache_items(items_for_remote_cache: Dict[str, Tuple[UserProfile]], user_profile: UserProfile) -> None:
    if False:
        i = 10
        return i + 15
    for api_key in get_all_api_keys(user_profile):
        items_for_remote_cache[user_profile_by_api_key_cache_key(api_key)] = (user_profile,)
    items_for_remote_cache[user_profile_cache_key(user_profile.email, user_profile.realm)] = (user_profile,)

def client_cache_items(items_for_remote_cache: Dict[str, Tuple[Client]], client: Client) -> None:
    if False:
        i = 10
        return i + 15
    items_for_remote_cache[get_client_cache_key(client.name)] = (client,)

def session_cache_items(items_for_remote_cache: Dict[str, Dict[str, object]], session: Session) -> None:
    if False:
        while True:
            i = 10
    if settings.SESSION_ENGINE != 'zerver.lib.safe_session_cached_db':
        return
    store = session_engine.SessionStore(session_key=session.session_key)
    assert isinstance(store, SessionStore)
    items_for_remote_cache[store.cache_key] = store.decode(session.session_data)

def get_active_realm_ids() -> ValuesQuerySet[RealmCount, int]:
    if False:
        for i in range(10):
            print('nop')
    'For installations like Zulip Cloud hosting a lot of realms, it only makes\n    sense to do cache-filling work for realms that have any currently\n    active users/clients.  Otherwise, we end up with every single-user\n    trial organization that has ever been created costing us N streams\n    worth of cache work (where N is the number of default streams for\n    a new organization).\n    '
    date = timezone_now() - datetime.timedelta(days=2)
    return RealmCount.objects.filter(end_time__gte=date, property='1day_actives::day', value__gt=0).distinct('realm_id').values_list('realm_id', flat=True)

def get_users() -> QuerySet[UserProfile]:
    if False:
        while True:
            i = 10
    return UserProfile.objects.select_related('realm', 'bot_owner').filter(long_term_idle=False, realm__in=get_active_realm_ids())
cache_fillers: Dict[str, Tuple[Callable[[], Iterable[Any]], Callable[[Dict[str, Any], Any], None], int, int]] = {'user': (get_users, user_cache_items, 3600 * 24 * 7, 10000), 'client': (lambda : Client.objects.all(), client_cache_items, 3600 * 24 * 7, 10000), 'session': (lambda : Session.objects.all(), session_cache_items, 3600 * 24 * 7, 10000)}

class SQLQueryCounter:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.count = 0

    def __call__(self, execute: Callable[[str, Any, bool, Dict[str, Any]], Any], sql: str, params: Any, many: bool, context: Dict[str, Any]) -> Any:
        if False:
            print('Hello World!')
        self.count += 1
        return execute(sql, params, many, context)

def fill_remote_cache(cache: str) -> None:
    if False:
        i = 10
        return i + 15
    remote_cache_time_start = get_remote_cache_time()
    remote_cache_requests_start = get_remote_cache_requests()
    items_for_remote_cache: Dict[str, Any] = {}
    (objects, items_filler, timeout, batch_size) = cache_fillers[cache]
    count = 0
    db_query_counter = SQLQueryCounter()
    with connection.execute_wrapper(db_query_counter):
        for obj in objects():
            items_filler(items_for_remote_cache, obj)
            count += 1
            if count % batch_size == 0:
                cache_set_many(items_for_remote_cache, timeout=3600 * 24)
                items_for_remote_cache = {}
        cache_set_many(items_for_remote_cache, timeout=3600 * 24 * 7)
    logging.info('Successfully populated %s cache: %d items, %d DB queries, %d memcached sets, %.2f seconds', cache, count, db_query_counter.count, get_remote_cache_requests() - remote_cache_requests_start, get_remote_cache_time() - remote_cache_time_start)