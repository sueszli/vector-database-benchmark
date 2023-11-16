import json
import os
import sys
from email.headerregistry import Address
from functools import wraps
from typing import Any, Callable, Dict, List, Set, TypeVar
from typing_extensions import ParamSpec
from zulip import Client
from zerver.models import get_realm, get_user
from zerver.openapi.openapi import validate_against_openapi_schema
ZULIP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_FUNCTIONS: Dict[str, Callable[..., object]] = {}
REGISTERED_TEST_FUNCTIONS: Set[str] = set()
CALLED_TEST_FUNCTIONS: Set[str] = set()
ParamT = ParamSpec('ParamT')
ReturnT = TypeVar('ReturnT')

def openapi_test_function(endpoint: str) -> Callable[[Callable[ParamT, ReturnT]], Callable[ParamT, ReturnT]]:
    if False:
        return 10
    'This decorator is used to register an OpenAPI test function with\n    its endpoint. Example usage:\n\n    @openapi_test_function("/messages/render:post")\n    def ...\n    '

    def wrapper(test_func: Callable[ParamT, ReturnT]) -> Callable[ParamT, ReturnT]:
        if False:
            return 10

        @wraps(test_func)
        def _record_calls_wrapper(*args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
            if False:
                return 10
            CALLED_TEST_FUNCTIONS.add(test_func.__name__)
            return test_func(*args, **kwargs)
        REGISTERED_TEST_FUNCTIONS.add(test_func.__name__)
        TEST_FUNCTIONS[endpoint] = _record_calls_wrapper
        return _record_calls_wrapper
    return wrapper

def ensure_users(ids_list: List[int], user_names: List[str]) -> None:
    if False:
        while True:
            i = 10
    realm = get_realm('zulip')
    user_ids = [get_user(Address(username=name, domain='zulip.com').addr_spec, realm).id for name in user_names]
    assert ids_list == user_ids

@openapi_test_function('/users/me/subscriptions:post')
def add_subscriptions(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.add_subscriptions(streams=[{'name': 'new stream', 'description': 'New stream for testing'}])
    validate_against_openapi_schema(result, '/users/me/subscriptions', 'post', '200')
    ensure_users([25], ['newbie'])
    user_id = 25
    result = client.add_subscriptions(streams=[{'name': 'new stream', 'description': 'New stream for testing'}], principals=[user_id])
    assert result['result'] == 'success'
    assert 'newbie@zulip.com' in result['subscribed']

def test_add_subscriptions_already_subscribed(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.add_subscriptions(streams=[{'name': 'new stream', 'description': 'New stream for testing'}], principals=['newbie@zulip.com'])
    validate_against_openapi_schema(result, '/users/me/subscriptions', 'post', '200')

def test_authorization_errors_fatal(client: Client, nonadmin_client: Client) -> None:
    if False:
        return 10
    client.add_subscriptions(streams=[{'name': 'private_stream'}])
    stream_id = client.get_stream_id('private_stream')['stream_id']
    client.call_endpoint(f'streams/{stream_id}', method='PATCH', request={'is_private': True})
    result = nonadmin_client.add_subscriptions(streams=[{'name': 'private_stream'}], authorization_errors_fatal=False)
    validate_against_openapi_schema(result, '/users/me/subscriptions', 'post', '400')
    result = nonadmin_client.add_subscriptions(streams=[{'name': 'private_stream'}], authorization_errors_fatal=True)
    validate_against_openapi_schema(result, '/users/me/subscriptions', 'post', '400')

@openapi_test_function('/realm/presence:get')
def get_presence(client: Client) -> None:
    if False:
        while True:
            i = 10
    result = client.get_realm_presence()
    validate_against_openapi_schema(result, '/realm/presence', 'get', '200')

@openapi_test_function('/default_streams:post')
def add_default_stream(client: Client) -> None:
    if False:
        print('Hello World!')
    stream_id = 7
    result = client.add_default_stream(stream_id)
    validate_against_openapi_schema(result, '/default_streams', 'post', '200')

@openapi_test_function('/default_streams:delete')
def remove_default_stream(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    request = {'stream_id': 7}
    result = client.call_endpoint(url='/default_streams', method='DELETE', request=request)
    validate_against_openapi_schema(result, '/default_streams', 'delete', '200')

@openapi_test_function('/users/{user_id_or_email}/presence:get')
def get_user_presence(client: Client) -> None:
    if False:
        return 10
    result = client.get_user_presence('iago@zulip.com')
    validate_against_openapi_schema(result, '/users/{user_id_or_email}/presence', 'get', '200')

@openapi_test_function('/users/me/presence:post')
def update_presence(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    request = {'status': 'active', 'ping_only': False, 'new_user_input': False}
    result = client.update_presence(request)
    assert result['result'] == 'success'

@openapi_test_function('/users:post')
def create_user(client: Client) -> None:
    if False:
        return 10
    request = {'email': 'newbie@zulip.com', 'password': 'temp', 'full_name': 'New User'}
    result = client.create_user(request)
    validate_against_openapi_schema(result, '/users', 'post', '200')
    result = client.create_user(request)
    validate_against_openapi_schema(result, '/users', 'post', '400')

@openapi_test_function('/users/me/status:post')
def update_status(client: Client) -> None:
    if False:
        while True:
            i = 10
    request = {'status_text': 'on vacation', 'away': False, 'emoji_name': 'car', 'emoji_code': '1f697', 'reaction_type': 'unicode_emoji'}
    result = client.call_endpoint(url='/users/me/status', method='POST', request=request)
    validate_against_openapi_schema(result, '/users/me/status', 'post', '200')
    request = {'status_text': 'This is a message that exceeds 60 characters, and so should throw an error.', 'away': 'false'}
    validate_against_openapi_schema(result, '/users/me/status', 'post', '400')

@openapi_test_function('/users:get')
def get_members(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.get_members()
    validate_against_openapi_schema(result, '/users', 'get', '200')
    members = [m for m in result['members'] if m['email'] == 'newbie@zulip.com']
    assert len(members) == 1
    newbie = members[0]
    assert not newbie['is_admin']
    assert newbie['full_name'] == 'New User'
    result = client.get_members({'client_gravatar': False})
    validate_against_openapi_schema(result, '/users', 'get', '200')
    assert result['members'][0]['avatar_url'] is not None
    result = client.get_members({'include_custom_profile_fields': True})
    validate_against_openapi_schema(result, '/users', 'get', '200')
    for member in result['members']:
        if member['is_bot']:
            assert member.get('profile_data', None) is None
        else:
            assert member.get('profile_data', None) is not None
        assert member['avatar_url'] is None

@openapi_test_function('/users/{email}:get')
def get_user_by_email(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    email = 'iago@zulip.com'
    result = client.call_endpoint(url=f'/users/{email}', method='GET')
    validate_against_openapi_schema(result, '/users/{email}', 'get', '200')

@openapi_test_function('/users/{user_id}:get')
def get_single_user(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    ensure_users([8], ['cordelia'])
    user_id = 8
    result = client.get_user_by_id(user_id)
    validate_against_openapi_schema(result, '/users/{user_id}', 'get', '200')
    result = client.get_user_by_id(user_id, include_custom_profile_fields=True)
    validate_against_openapi_schema(result, '/users/{user_id}', 'get', '200')

@openapi_test_function('/users/{user_id}:delete')
def deactivate_user(client: Client) -> None:
    if False:
        return 10
    ensure_users([8], ['cordelia'])
    user_id = 8
    result = client.deactivate_user_by_id(user_id)
    validate_against_openapi_schema(result, '/users/{user_id}', 'delete', '200')

@openapi_test_function('/users/{user_id}/reactivate:post')
def reactivate_user(client: Client) -> None:
    if False:
        return 10
    user_id = 8
    result = client.reactivate_user_by_id(user_id)
    validate_against_openapi_schema(result, '/users/{user_id}/reactivate', 'post', '200')

@openapi_test_function('/users/{user_id}:patch')
def update_user(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    ensure_users([8, 10], ['cordelia', 'hamlet'])
    user_id = 10
    result = client.update_user_by_id(user_id, full_name='New Name')
    validate_against_openapi_schema(result, '/users/{user_id}', 'patch', '200')
    user_id = 8
    result = client.update_user_by_id(user_id, profile_data=[{'id': 9, 'value': 'some data'}])
    validate_against_openapi_schema(result, '/users/{user_id}', 'patch', '400')

@openapi_test_function('/users/{user_id}/subscriptions/{stream_id}:get')
def get_subscription_status(client: Client) -> None:
    if False:
        while True:
            i = 10
    ensure_users([7], ['zoe'])
    user_id = 7
    stream_id = 1
    result = client.call_endpoint(url=f'/users/{user_id}/subscriptions/{stream_id}', method='GET')
    validate_against_openapi_schema(result, '/users/{user_id}/subscriptions/{stream_id}', 'get', '200')

@openapi_test_function('/realm/linkifiers:get')
def get_realm_linkifiers(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.call_endpoint(url='/realm/linkifiers', method='GET')
    validate_against_openapi_schema(result, '/realm/linkifiers', 'get', '200')

@openapi_test_function('/realm/linkifiers:patch')
def reorder_realm_linkifiers(client: Client) -> None:
    if False:
        while True:
            i = 10
    order = [4, 3, 2, 1]
    request = {'ordered_linkifier_ids': json.dumps(order)}
    result = client.call_endpoint(url='/realm/linkifiers', method='PATCH', request=request)
    validate_against_openapi_schema(result, '/realm/linkifiers', 'patch', '200')

@openapi_test_function('/realm/profile_fields:get')
def get_realm_profile_fields(client: Client) -> None:
    if False:
        while True:
            i = 10
    result = client.call_endpoint(url='/realm/profile_fields', method='GET')
    validate_against_openapi_schema(result, '/realm/profile_fields', 'get', '200')

@openapi_test_function('/realm/profile_fields:patch')
def reorder_realm_profile_fields(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    order = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    request = {'order': json.dumps(order)}
    result = client.call_endpoint(url='/realm/profile_fields', method='PATCH', request=request)
    validate_against_openapi_schema(result, '/realm/profile_fields', 'patch', '200')

@openapi_test_function('/realm/profile_fields:post')
def create_realm_profile_field(client: Client) -> None:
    if False:
        while True:
            i = 10
    request = {'name': 'Phone', 'hint': 'Contact no.', 'field_type': 1}
    result = client.call_endpoint(url='/realm/profile_fields', method='POST', request=request)
    validate_against_openapi_schema(result, '/realm/profile_fields', 'post', '200')

@openapi_test_function('/realm/filters:post')
def add_realm_filter(client: Client) -> None:
    if False:
        while True:
            i = 10
    request = {'pattern': '#(?P<id>[0-9]+)', 'url_template': 'https://github.com/zulip/zulip/issues/{id}'}
    result = client.call_endpoint('/realm/filters', method='POST', request=request)
    validate_against_openapi_schema(result, '/realm/filters', 'post', '200')

@openapi_test_function('/realm/filters/{filter_id}:patch')
def update_realm_filter(client: Client) -> None:
    if False:
        print('Hello World!')
    filter_id = 4
    request = {'pattern': '#(?P<id>[0-9]+)', 'url_template': 'https://github.com/zulip/zulip/issues/{id}'}
    result = client.call_endpoint(url=f'/realm/filters/{filter_id}', method='PATCH', request=request)
    validate_against_openapi_schema(result, '/realm/filters/{filter_id}', 'patch', '200')

@openapi_test_function('/realm/filters/{filter_id}:delete')
def remove_realm_filter(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.remove_realm_filter(4)
    validate_against_openapi_schema(result, '/realm/filters/{filter_id}', 'delete', '200')

@openapi_test_function('/realm/playgrounds:post')
def add_realm_playground(client: Client) -> None:
    if False:
        return 10
    request = {'name': 'Python playground', 'pygments_language': 'Python', 'url_template': 'https://python.example.com?code={code}'}
    result = client.call_endpoint(url='/realm/playgrounds', method='POST', request=request)
    validate_against_openapi_schema(result, '/realm/playgrounds', 'post', '200')

@openapi_test_function('/realm/playgrounds/{playground_id}:delete')
def remove_realm_playground(client: Client) -> None:
    if False:
        while True:
            i = 10
    result = client.call_endpoint(url='/realm/playgrounds/1', method='DELETE')
    validate_against_openapi_schema(result, '/realm/playgrounds/{playground_id}', 'delete', '200')

@openapi_test_function('/users/me:get')
def get_profile(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    result = client.get_profile()
    validate_against_openapi_schema(result, '/users/me', 'get', '200')

@openapi_test_function('/users/me:delete')
def deactivate_own_user(client: Client, owner_client: Client) -> None:
    if False:
        while True:
            i = 10
    user_id = client.get_profile()['user_id']
    result = client.call_endpoint(url='/users/me', method='DELETE')
    owner_client.reactivate_user_by_id(user_id)
    validate_against_openapi_schema(result, '/users/me', 'delete', '200')

@openapi_test_function('/get_stream_id:get')
def get_stream_id(client: Client) -> int:
    if False:
        print('Hello World!')
    stream_name = 'new stream'
    result = client.get_stream_id(stream_name)
    validate_against_openapi_schema(result, '/get_stream_id', 'get', '200')
    return result['stream_id']

@openapi_test_function('/streams/{stream_id}:delete')
def archive_stream(client: Client, stream_id: int) -> None:
    if False:
        while True:
            i = 10
    result = client.add_subscriptions(streams=[{'name': 'stream to be archived', 'description': 'New stream for testing'}])
    stream_id = client.get_stream_id('stream to be archived')['stream_id']
    result = client.delete_stream(stream_id)
    validate_against_openapi_schema(result, '/streams/{stream_id}', 'delete', '200')
    assert result['result'] == 'success'

@openapi_test_function('/streams/{stream_id}/delete_topic:post')
def delete_topic(client: Client, stream_id: int, topic: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    request = {'topic_name': topic}
    result = client.call_endpoint(url=f'/streams/{stream_id}/delete_topic', method='POST', request=request)
    validate_against_openapi_schema(result, '/streams/{stream_id}/delete_topic', 'post', '200')
    assert result['result'] == 'success'

@openapi_test_function('/streams:get')
def get_streams(client: Client) -> None:
    if False:
        return 10
    result = client.get_streams()
    validate_against_openapi_schema(result, '/streams', 'get', '200')
    streams = [s for s in result['streams'] if s['name'] == 'new stream']
    assert streams[0]['description'] == 'New stream for testing'
    result = client.get_streams(include_public=False)
    validate_against_openapi_schema(result, '/streams', 'get', '200')
    assert len(result['streams']) == 5

@openapi_test_function('/streams/{stream_id}:patch')
def update_stream(client: Client, stream_id: int) -> None:
    if False:
        print('Hello World!')
    request = {'stream_id': stream_id, 'stream_post_policy': 2, 'is_private': True}
    result = client.update_stream(request)
    validate_against_openapi_schema(result, '/streams/{stream_id}', 'patch', '200')
    assert result['result'] == 'success'

@openapi_test_function('/user_groups:get')
def get_user_groups(client: Client) -> int:
    if False:
        print('Hello World!')
    result = client.get_user_groups()
    validate_against_openapi_schema(result, '/user_groups', 'get', '200')
    [hamlet_user_group] = (u for u in result['user_groups'] if u['name'] == 'hamletcharacters')
    assert hamlet_user_group['description'] == 'Characters of Hamlet'
    [marketing_user_group] = (u for u in result['user_groups'] if u['name'] == 'marketing')
    return marketing_user_group['id']

def test_user_not_authorized_error(nonadmin_client: Client) -> None:
    if False:
        print('Hello World!')
    result = nonadmin_client.get_streams(include_all_active=True)
    validate_against_openapi_schema(result, '/rest-error-handling', 'post', '400')

@openapi_test_function('/streams/{stream_id}/members:get')
def get_subscribers(client: Client) -> None:
    if False:
        while True:
            i = 10
    ensure_users([11, 25], ['iago', 'newbie'])
    result = client.get_subscribers(stream='new stream')
    assert result['subscribers'] == [11, 25]

def get_user_agent(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = client.get_user_agent()
    assert result.startswith('ZulipPython/')

@openapi_test_function('/users/me/subscriptions:get')
def get_subscriptions(client: Client) -> None:
    if False:
        return 10
    result = client.get_subscriptions()
    validate_against_openapi_schema(result, '/users/me/subscriptions', 'get', '200')
    streams = [s for s in result['subscriptions'] if s['name'] == 'new stream']
    assert streams[0]['description'] == 'New stream for testing'

@openapi_test_function('/users/me/subscriptions:delete')
def remove_subscriptions(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    result = client.remove_subscriptions(['new stream'])
    validate_against_openapi_schema(result, '/users/me/subscriptions', 'delete', '200')
    result = client.get_subscriptions()
    assert result['result'] == 'success'
    streams = [s for s in result['subscriptions'] if s['name'] == 'new stream']
    assert len(streams) == 0
    result = client.remove_subscriptions(['new stream'], principals=['newbie@zulip.com'])
    validate_against_openapi_schema(result, '/users/me/subscriptions', 'delete', '200')

@openapi_test_function('/users/me/subscriptions/muted_topics:patch')
def toggle_mute_topic(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    message = {'type': 'stream', 'to': 'Denmark', 'topic': 'boat party'}
    client.call_endpoint(url='messages', method='POST', request=message)
    request = {'stream': 'Denmark', 'topic': 'boat party', 'op': 'add'}
    result = client.mute_topic(request)
    validate_against_openapi_schema(result, '/users/me/subscriptions/muted_topics', 'patch', '200')
    request = {'stream': 'Denmark', 'topic': 'boat party', 'op': 'remove'}
    result = client.mute_topic(request)
    validate_against_openapi_schema(result, '/users/me/subscriptions/muted_topics', 'patch', '200')

@openapi_test_function('/user_topics:post')
def update_user_topic(client: Client) -> None:
    if False:
        print('Hello World!')
    stream_id = client.get_stream_id('Denmark')['stream_id']
    request = {'stream_id': stream_id, 'topic': 'dinner', 'visibility_policy': 1}
    result = client.call_endpoint(url='user_topics', method='POST', request=request)
    validate_against_openapi_schema(result, '/user_topics', 'post', '200')
    request = {'stream_id': stream_id, 'topic': 'dinner', 'visibility_policy': 0}
    result = client.call_endpoint(url='user_topics', method='POST', request=request)
    validate_against_openapi_schema(result, '/user_topics', 'post', '200')

@openapi_test_function('/users/me/muted_users/{muted_user_id}:post')
def add_user_mute(client: Client) -> None:
    if False:
        print('Hello World!')
    ensure_users([10], ['hamlet'])
    muted_user_id = 10
    result = client.call_endpoint(url=f'/users/me/muted_users/{muted_user_id}', method='POST')
    validate_against_openapi_schema(result, '/users/me/muted_users/{muted_user_id}', 'post', '200')

@openapi_test_function('/users/me/muted_users/{muted_user_id}:delete')
def remove_user_mute(client: Client) -> None:
    if False:
        while True:
            i = 10
    ensure_users([10], ['hamlet'])
    muted_user_id = 10
    result = client.call_endpoint(url=f'/users/me/muted_users/{muted_user_id}', method='DELETE')
    validate_against_openapi_schema(result, '/users/me/muted_users/{muted_user_id}', 'delete', '200')

@openapi_test_function('/mark_all_as_read:post')
def mark_all_as_read(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.mark_all_as_read()
    validate_against_openapi_schema(result, '/mark_all_as_read', 'post', '200')

@openapi_test_function('/mark_stream_as_read:post')
def mark_stream_as_read(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    result = client.mark_stream_as_read(1)
    validate_against_openapi_schema(result, '/mark_stream_as_read', 'post', '200')

@openapi_test_function('/mark_topic_as_read:post')
def mark_topic_as_read(client: Client) -> None:
    if False:
        print('Hello World!')
    topic_name = client.get_stream_topics(1)['topics'][0]['name']
    result = client.mark_topic_as_read(1, topic_name)
    validate_against_openapi_schema(result, '/mark_stream_as_read', 'post', '200')

@openapi_test_function('/users/me/subscriptions/properties:post')
def update_subscription_settings(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    request = [{'stream_id': 1, 'property': 'pin_to_top', 'value': True}, {'stream_id': 7, 'property': 'color', 'value': '#f00f00'}]
    result = client.update_subscription_settings(request)
    validate_against_openapi_schema(result, '/users/me/subscriptions/properties', 'POST', '200')

@openapi_test_function('/messages/render:post')
def render_message(client: Client) -> None:
    if False:
        print('Hello World!')
    request = {'content': '**foo**'}
    result = client.render_message(request)
    validate_against_openapi_schema(result, '/messages/render', 'post', '200')

@openapi_test_function('/messages:get')
def get_messages(client: Client) -> None:
    if False:
        print('Hello World!')
    request: Dict[str, Any] = {'anchor': 'newest', 'num_before': 100, 'num_after': 0, 'narrow': [{'operator': 'sender', 'operand': 'iago@zulip.com'}, {'operator': 'stream', 'operand': 'Verona'}]}
    result = client.get_messages(request)
    validate_against_openapi_schema(result, '/messages', 'get', '200')
    assert len(result['messages']) <= request['num_before']

@openapi_test_function('/messages/matches_narrow:get')
def check_messages_match_narrow(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    message = {'type': 'stream', 'to': 'Verona', 'topic': 'test_topic', 'content': 'http://foo.com'}
    msg_ids = []
    response = client.send_message(message)
    msg_ids.append(response['id'])
    message['content'] = 'no link here'
    response = client.send_message(message)
    msg_ids.append(response['id'])
    request = {'msg_ids': msg_ids, 'narrow': [{'operator': 'has', 'operand': 'link'}]}
    result = client.call_endpoint(url='messages/matches_narrow', method='GET', request=request)
    validate_against_openapi_schema(result, '/messages/matches_narrow', 'get', '200')

@openapi_test_function('/messages/{message_id}:get')
def get_raw_message(client: Client, message_id: int) -> None:
    if False:
        i = 10
        return i + 15
    assert int(message_id)
    result = client.get_raw_message(message_id)
    validate_against_openapi_schema(result, '/messages/{message_id}', 'get', '200')

@openapi_test_function('/attachments:get')
def get_attachments(client: Client) -> int:
    if False:
        return 10
    result = client.get_attachments()
    validate_against_openapi_schema(result, '/attachments', 'get', '200')
    return result['attachments'][0]['id']

@openapi_test_function('/attachments/{attachment_id}:delete')
def remove_attachment(client: Client, attachment_id: int) -> None:
    if False:
        i = 10
        return i + 15
    url = 'attachments/' + str(attachment_id)
    result = client.call_endpoint(url=url, method='DELETE')
    validate_against_openapi_schema(result, '/attachments/{attachment_id}', 'delete', '200')

@openapi_test_function('/messages:post')
def send_message(client: Client) -> int:
    if False:
        while True:
            i = 10
    request: Dict[str, Any] = {}
    request = {'type': 'stream', 'to': 'Denmark', 'topic': 'Castle', 'content': 'I come not, friends, to steal away your hearts.'}
    result = client.send_message(request)
    validate_against_openapi_schema(result, '/messages', 'post', '200')
    message_id = result['id']
    url = 'messages/' + str(message_id)
    result = client.call_endpoint(url=url, method='GET')
    assert result['result'] == 'success'
    assert result['raw_content'] == request['content']
    ensure_users([10], ['hamlet'])
    user_id = 10
    request = {'type': 'private', 'to': [user_id], 'content': 'With mirth and laughter let old wrinkles come.'}
    result = client.send_message(request)
    validate_against_openapi_schema(result, '/messages', 'post', '200')
    message_id = result['id']
    url = 'messages/' + str(message_id)
    result = client.call_endpoint(url=url, method='GET')
    assert result['result'] == 'success'
    assert result['raw_content'] == request['content']
    return message_id

@openapi_test_function('/messages/{message_id}/reactions:post')
def add_reaction(client: Client, message_id: int) -> None:
    if False:
        return 10
    request: Dict[str, Any] = {}
    request = {'message_id': message_id, 'emoji_name': 'octopus'}
    result = client.add_reaction(request)
    validate_against_openapi_schema(result, '/messages/{message_id}/reactions', 'post', '200')

@openapi_test_function('/messages/{message_id}/reactions:delete')
def remove_reaction(client: Client, message_id: int) -> None:
    if False:
        return 10
    request: Dict[str, Any] = {}
    request = {'message_id': message_id, 'emoji_name': 'octopus'}
    result = client.remove_reaction(request)
    validate_against_openapi_schema(result, '/messages/{message_id}/reactions', 'delete', '200')

@openapi_test_function('/messages/{message_id}/read_receipts:get')
def get_read_receipts(client: Client, message_id: int) -> None:
    if False:
        print('Hello World!')
    result = client.call_endpoint(f'/messages/{message_id}/read_receipts', method='GET')
    validate_against_openapi_schema(result, '/messages/{message_id}/read_receipts', 'get', '200')

def test_nonexistent_stream_error(client: Client) -> None:
    if False:
        while True:
            i = 10
    request = {'type': 'stream', 'to': 'nonexistent_stream', 'topic': 'Castle', 'content': 'I come not, friends, to steal away your hearts.'}
    result = client.send_message(request)
    validate_against_openapi_schema(result, '/messages', 'post', '400')

def test_private_message_invalid_recipient(client: Client) -> None:
    if False:
        print('Hello World!')
    request = {'type': 'private', 'to': 'eeshan@zulip.com', 'content': 'With mirth and laughter let old wrinkles come.'}
    result = client.send_message(request)
    validate_against_openapi_schema(result, '/messages', 'post', '400')

@openapi_test_function('/messages/{message_id}:patch')
def update_message(client: Client, message_id: int) -> None:
    if False:
        while True:
            i = 10
    assert int(message_id)
    request = {'message_id': message_id, 'content': 'New content'}
    result = client.update_message(request)
    validate_against_openapi_schema(result, '/messages/{message_id}', 'patch', '200')
    url = 'messages/' + str(message_id)
    result = client.call_endpoint(url=url, method='GET')
    assert result['result'] == 'success'
    assert result['raw_content'] == request['content']

def test_update_message_edit_permission_error(client: Client, nonadmin_client: Client) -> None:
    if False:
        while True:
            i = 10
    request = {'type': 'stream', 'to': 'Denmark', 'topic': 'Castle', 'content': 'I come not, friends, to steal away your hearts.'}
    result = client.send_message(request)
    request = {'message_id': result['id'], 'content': 'New content'}
    result = nonadmin_client.update_message(request)
    validate_against_openapi_schema(result, '/messages/{message_id}', 'patch', '400')

@openapi_test_function('/messages/{message_id}:delete')
def delete_message(client: Client, message_id: int) -> None:
    if False:
        i = 10
        return i + 15
    result = client.delete_message(message_id)
    validate_against_openapi_schema(result, '/messages/{message_id}', 'delete', '200')

def test_delete_message_edit_permission_error(client: Client, nonadmin_client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    request = {'type': 'stream', 'to': 'Denmark', 'topic': 'Castle', 'content': 'I come not, friends, to steal away your hearts.'}
    result = client.send_message(request)
    result = nonadmin_client.delete_message(result['id'])
    validate_against_openapi_schema(result, '/messages/{message_id}', 'delete', '400')

@openapi_test_function('/messages/{message_id}/history:get')
def get_message_history(client: Client, message_id: int) -> None:
    if False:
        return 10
    result = client.get_message_history(message_id)
    validate_against_openapi_schema(result, '/messages/{message_id}/history', 'get', '200')

@openapi_test_function('/realm/emoji:get')
def get_realm_emoji(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.get_realm_emoji()
    validate_against_openapi_schema(result, '/realm/emoji', 'GET', '200')

@openapi_test_function('/messages/flags:post')
def update_message_flags(client: Client) -> None:
    if False:
        return 10
    request: Dict[str, Any] = {'type': 'stream', 'to': 'Denmark', 'topic': 'Castle', 'content': 'I come not, friends, to steal away your hearts.'}
    message_ids = [client.send_message(request)['id'] for i in range(3)]
    request = {'messages': message_ids, 'op': 'add', 'flag': 'read'}
    result = client.update_message_flags(request)
    validate_against_openapi_schema(result, '/messages/flags', 'post', '200')
    request = {'messages': message_ids, 'op': 'remove', 'flag': 'starred'}
    result = client.update_message_flags(request)
    validate_against_openapi_schema(result, '/messages/flags', 'post', '200')

def register_queue_all_events(client: Client) -> str:
    if False:
        for i in range(10):
            print('nop')
    result = client.register()
    validate_against_openapi_schema(result, '/register', 'post', '200')
    return result['queue_id']

@openapi_test_function('/register:post')
def register_queue(client: Client) -> str:
    if False:
        for i in range(10):
            print('nop')
    result = client.register(event_types=['message', 'realm_emoji'])
    validate_against_openapi_schema(result, '/register', 'post', '200')
    return result['queue_id']

@openapi_test_function('/events:delete')
def deregister_queue(client: Client, queue_id: str) -> None:
    if False:
        i = 10
        return i + 15
    result = client.deregister(queue_id)
    validate_against_openapi_schema(result, '/events', 'delete', '200')
    result = client.deregister(queue_id)
    validate_against_openapi_schema(result, '/events', 'delete', '400')

@openapi_test_function('/events:get')
def get_queue(client: Client, queue_id: str) -> None:
    if False:
        while True:
            i = 10
    result = client.get_events(queue_id=queue_id, last_event_id=-1)
    validate_against_openapi_schema(result, '/events', 'get', '200')

@openapi_test_function('/server_settings:get')
def get_server_settings(client: Client) -> None:
    if False:
        return 10
    result = client.get_server_settings()
    validate_against_openapi_schema(result, '/server_settings', 'get', '200')

@openapi_test_function('/settings:patch')
def update_settings(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    request = {'enable_offline_push_notifications': True, 'enable_online_push_notifications': True, 'emojiset': 'google'}
    result = client.call_endpoint('/settings', method='PATCH', request=request)
    validate_against_openapi_schema(result, '/settings', 'patch', '200')

@openapi_test_function('/user_uploads:post')
def upload_file(client: Client) -> None:
    if False:
        print('Hello World!')
    path_to_file = os.path.join(ZULIP_DIR, 'zerver', 'tests', 'images', 'img.jpg')
    with open(path_to_file, 'rb') as fp:
        result = client.upload_file(fp)
    client.send_message({'type': 'stream', 'to': 'Denmark', 'topic': 'Castle', 'content': 'Check out [this picture]({}) of my castle!'.format(result['uri'])})
    validate_against_openapi_schema(result, '/user_uploads', 'post', '200')

@openapi_test_function('/users/me/{stream_id}/topics:get')
def get_stream_topics(client: Client, stream_id: int) -> None:
    if False:
        i = 10
        return i + 15
    result = client.get_stream_topics(stream_id)
    validate_against_openapi_schema(result, '/users/me/{stream_id}/topics', 'get', '200')

@openapi_test_function('/typing:post')
def set_typing_status(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    ensure_users([10, 11], ['hamlet', 'iago'])
    user_id1 = 10
    user_id2 = 11
    request = {'op': 'start', 'to': [user_id1, user_id2]}
    result = client.set_typing_status(request)
    validate_against_openapi_schema(result, '/typing', 'post', '200')
    user_id1 = 10
    user_id2 = 11
    request = {'op': 'stop', 'to': [user_id1, user_id2]}
    result = client.set_typing_status(request)
    validate_against_openapi_schema(result, '/typing', 'post', '200')
    stream_id = client.get_stream_id('Denmark')['stream_id']
    topic = 'typing status'
    request = {'type': 'stream', 'op': 'start', 'stream_id': stream_id, 'topic': topic}
    result = client.set_typing_status(request)
    validate_against_openapi_schema(result, '/typing', 'post', '200')
    stream_id = client.get_stream_id('Denmark')['stream_id']
    topic = 'typing status'
    request = {'type': 'stream', 'op': 'stop', 'stream_id': stream_id, 'topic': topic}
    result = client.set_typing_status(request)
    validate_against_openapi_schema(result, '/typing', 'post', '200')

@openapi_test_function('/realm/emoji/{emoji_name}:post')
def upload_custom_emoji(client: Client) -> None:
    if False:
        print('Hello World!')
    emoji_path = os.path.join(ZULIP_DIR, 'zerver', 'tests', 'images', 'img.jpg')
    with open(emoji_path, 'rb') as fp:
        emoji_name = 'my_custom_emoji'
        result = client.call_endpoint(f'realm/emoji/{emoji_name}', method='POST', files=[fp])
    validate_against_openapi_schema(result, '/realm/emoji/{emoji_name}', 'post', '200')

@openapi_test_function('/realm/emoji/{emoji_name}:delete')
def delete_custom_emoji(client: Client) -> None:
    if False:
        return 10
    emoji_name = 'my_custom_emoji'
    result = client.call_endpoint(f'realm/emoji/{emoji_name}', method='DELETE')
    validate_against_openapi_schema(result, '/realm/emoji/{emoji_name}', 'delete', '200')

@openapi_test_function('/users/me/alert_words:get')
def get_alert_words(client: Client) -> None:
    if False:
        print('Hello World!')
    result = client.get_alert_words()
    validate_against_openapi_schema(result, '/users/me/alert_words', 'get', '200')
    assert result['result'] == 'success'

@openapi_test_function('/users/me/alert_words:post')
def add_alert_words(client: Client) -> None:
    if False:
        return 10
    word = ['foo', 'bar']
    result = client.add_alert_words(word)
    validate_against_openapi_schema(result, '/users/me/alert_words', 'post', '200')
    assert result['result'] == 'success'

@openapi_test_function('/users/me/alert_words:delete')
def remove_alert_words(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    word = ['foo']
    result = client.remove_alert_words(word)
    validate_against_openapi_schema(result, '/users/me/alert_words', 'delete', '200')
    assert result['result'] == 'success'

@openapi_test_function('/user_groups/create:post')
def create_user_group(client: Client) -> None:
    if False:
        while True:
            i = 10
    ensure_users([6, 7, 8, 10], ['aaron', 'zoe', 'cordelia', 'hamlet'])
    request = {'name': 'marketing', 'description': 'The marketing team.', 'members': [6, 7, 8, 10]}
    result = client.create_user_group(request)
    validate_against_openapi_schema(result, '/user_groups/create', 'post', '200')
    assert result['result'] == 'success'

@openapi_test_function('/user_groups/{user_group_id}:patch')
def update_user_group(client: Client, user_group_id: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    request = {'group_id': user_group_id, 'name': 'marketing', 'description': 'The marketing team.'}
    result = client.update_user_group(request)
    assert result['result'] == 'success'

@openapi_test_function('/user_groups/{user_group_id}:delete')
def remove_user_group(client: Client, user_group_id: int) -> None:
    if False:
        print('Hello World!')
    result = client.remove_user_group(user_group_id)
    validate_against_openapi_schema(result, '/user_groups/{user_group_id}', 'delete', '200')
    assert result['result'] == 'success'

@openapi_test_function('/user_groups/{user_group_id}/members:post')
def update_user_group_members(client: Client, user_group_id: int) -> None:
    if False:
        print('Hello World!')
    ensure_users([8, 10, 11], ['cordelia', 'hamlet', 'iago'])
    request = {'delete': [8, 10], 'add': [11]}
    result = client.update_user_group_members(user_group_id, request)
    validate_against_openapi_schema(result, '/user_groups/{group_id}/members', 'post', '200')

def test_invalid_api_key(client_with_invalid_key: Client) -> None:
    if False:
        i = 10
        return i + 15
    result = client_with_invalid_key.get_subscriptions()
    validate_against_openapi_schema(result, '/rest-error-handling', 'post', '400')

def test_missing_request_argument(client: Client) -> None:
    if False:
        return 10
    result = client.render_message({})
    validate_against_openapi_schema(result, '/rest-error-handling', 'post', '400')

def test_user_account_deactivated(client: Client) -> None:
    if False:
        while True:
            i = 10
    request = {'content': '**foo**'}
    result = client.render_message(request)
    validate_against_openapi_schema(result, '/rest-error-handling', 'post', '403')

def test_realm_deactivated(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    request = {'content': '**foo**'}
    result = client.render_message(request)
    validate_against_openapi_schema(result, '/rest-error-handling', 'post', '403')

def test_invalid_stream_error(client: Client) -> None:
    if False:
        while True:
            i = 10
    result = client.get_stream_id('nonexistent')
    validate_against_openapi_schema(result, '/get_stream_id', 'get', '400')

def test_messages(client: Client, nonadmin_client: Client) -> None:
    if False:
        while True:
            i = 10
    render_message(client)
    message_id = send_message(client)
    add_reaction(client, message_id)
    remove_reaction(client, message_id)
    update_message(client, message_id)
    get_raw_message(client, message_id)
    get_messages(client)
    check_messages_match_narrow(client)
    get_message_history(client, message_id)
    get_read_receipts(client, message_id)
    delete_message(client, message_id)
    mark_all_as_read(client)
    mark_stream_as_read(client)
    mark_topic_as_read(client)
    update_message_flags(client)
    test_nonexistent_stream_error(client)
    test_private_message_invalid_recipient(client)
    test_update_message_edit_permission_error(client, nonadmin_client)
    test_delete_message_edit_permission_error(client, nonadmin_client)

def test_users(client: Client, owner_client: Client) -> None:
    if False:
        print('Hello World!')
    create_user(client)
    get_members(client)
    get_single_user(client)
    deactivate_user(client)
    reactivate_user(client)
    update_user(client)
    update_status(client)
    get_user_by_email(client)
    get_subscription_status(client)
    get_profile(client)
    update_settings(client)
    upload_file(client)
    attachment_id = get_attachments(client)
    remove_attachment(client, attachment_id)
    set_typing_status(client)
    update_presence(client)
    get_user_presence(client)
    get_presence(client)
    create_user_group(client)
    user_group_id = get_user_groups(client)
    update_user_group(client, user_group_id)
    update_user_group_members(client, user_group_id)
    remove_user_group(client, user_group_id)
    get_alert_words(client)
    add_alert_words(client)
    remove_alert_words(client)
    deactivate_own_user(client, owner_client)
    add_user_mute(client)
    remove_user_mute(client)
    get_alert_words(client)
    add_alert_words(client)
    remove_alert_words(client)

def test_streams(client: Client, nonadmin_client: Client) -> None:
    if False:
        while True:
            i = 10
    add_subscriptions(client)
    test_add_subscriptions_already_subscribed(client)
    get_subscriptions(client)
    stream_id = get_stream_id(client)
    update_stream(client, stream_id)
    get_streams(client)
    get_subscribers(client)
    remove_subscriptions(client)
    toggle_mute_topic(client)
    update_user_topic(client)
    update_subscription_settings(client)
    get_stream_topics(client, 1)
    delete_topic(client, 1, 'test')
    archive_stream(client, stream_id)
    add_default_stream(client)
    remove_default_stream(client)
    test_user_not_authorized_error(nonadmin_client)
    test_authorization_errors_fatal(client, nonadmin_client)

def test_queues(client: Client) -> None:
    if False:
        for i in range(10):
            print('nop')
    queue_id = register_queue(client)
    get_queue(client, queue_id)
    deregister_queue(client, queue_id)
    register_queue_all_events(client)

def test_server_organizations(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    get_realm_linkifiers(client)
    add_realm_filter(client)
    update_realm_filter(client)
    add_realm_playground(client)
    get_server_settings(client)
    reorder_realm_linkifiers(client)
    remove_realm_filter(client)
    remove_realm_playground(client)
    get_realm_emoji(client)
    upload_custom_emoji(client)
    delete_custom_emoji(client)
    get_realm_profile_fields(client)
    reorder_realm_profile_fields(client)
    create_realm_profile_field(client)

def test_errors(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    test_missing_request_argument(client)
    test_invalid_stream_error(client)

def test_the_api(client: Client, nonadmin_client: Client, owner_client: Client) -> None:
    if False:
        i = 10
        return i + 15
    get_user_agent(client)
    test_users(client, owner_client)
    test_streams(client, nonadmin_client)
    test_messages(client, nonadmin_client)
    test_queues(client)
    test_server_organizations(client)
    test_errors(client)
    sys.stdout.flush()
    if REGISTERED_TEST_FUNCTIONS != CALLED_TEST_FUNCTIONS:
        print('Error!  Some @openapi_test_function tests were never called:')
        print('  ', REGISTERED_TEST_FUNCTIONS - CALLED_TEST_FUNCTIONS)
        sys.exit(1)