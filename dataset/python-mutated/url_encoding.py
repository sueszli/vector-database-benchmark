from typing import Any, Dict, List
from urllib.parse import quote, urlsplit
import re2
from zerver.lib.topic import get_topic_from_message_info
from zerver.lib.types import UserDisplayRecipient
from zerver.models import Realm, Stream, UserProfile

def hash_util_encode(string: str) -> str:
    if False:
        while True:
            i = 10
    return quote(string, safe=b'').replace('.', '%2E').replace('%', '.')

def encode_stream(stream_id: int, stream_name: str) -> str:
    if False:
        while True:
            i = 10
    stream_name = stream_name.replace(' ', '-')
    return str(stream_id) + '-' + hash_util_encode(stream_name)

def personal_narrow_url(*, realm: Realm, sender: UserProfile) -> str:
    if False:
        return 10
    base_url = f'{realm.uri}/#narrow/dm/'
    encoded_user_name = re2.sub('[ "%\\/<>`\\p{C}]+', '-', sender.full_name)
    pm_slug = str(sender.id) + '-' + encoded_user_name
    return base_url + pm_slug

def huddle_narrow_url(*, user: UserProfile, display_recipient: List[UserDisplayRecipient]) -> str:
    if False:
        while True:
            i = 10
    realm = user.realm
    other_user_ids = [r['id'] for r in display_recipient if r['id'] != user.id]
    pm_slug = ','.join((str(user_id) for user_id in sorted(other_user_ids))) + '-group'
    base_url = f'{realm.uri}/#narrow/dm/'
    return base_url + pm_slug

def stream_narrow_url(realm: Realm, stream: Stream) -> str:
    if False:
        while True:
            i = 10
    base_url = f'{realm.uri}/#narrow/stream/'
    return base_url + encode_stream(stream.id, stream.name)

def topic_narrow_url(*, realm: Realm, stream: Stream, topic: str) -> str:
    if False:
        print('Hello World!')
    base_url = f'{realm.uri}/#narrow/stream/'
    return f'{base_url}{encode_stream(stream.id, stream.name)}/topic/{hash_util_encode(topic)}'

def near_message_url(realm: Realm, message: Dict[str, Any]) -> str:
    if False:
        return 10
    if message['type'] == 'stream':
        url = near_stream_message_url(realm=realm, message=message)
        return url
    url = near_pm_message_url(realm=realm, message=message)
    return url

def near_stream_message_url(realm: Realm, message: Dict[str, Any]) -> str:
    if False:
        while True:
            i = 10
    message_id = str(message['id'])
    stream_id = message['stream_id']
    stream_name = message['display_recipient']
    topic_name = get_topic_from_message_info(message)
    encoded_topic = hash_util_encode(topic_name)
    encoded_stream = encode_stream(stream_id=stream_id, stream_name=stream_name)
    parts = [realm.uri, '#narrow', 'stream', encoded_stream, 'topic', encoded_topic, 'near', message_id]
    full_url = '/'.join(parts)
    return full_url

def near_pm_message_url(realm: Realm, message: Dict[str, Any]) -> str:
    if False:
        for i in range(10):
            print('nop')
    message_id = str(message['id'])
    str_user_ids = [str(recipient['id']) for recipient in message['display_recipient']]
    pm_str = ','.join(str_user_ids) + '-pm'
    parts = [realm.uri, '#narrow', 'dm', pm_str, 'near', message_id]
    full_url = '/'.join(parts)
    return full_url

def append_url_query_string(original_url: str, query: str) -> str:
    if False:
        while True:
            i = 10
    u = urlsplit(original_url)
    query = u.query + ('&' if u.query and query else '') + query
    return u._replace(query=query).geturl()