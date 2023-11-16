from typing import Any, Dict
import orjson
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.user_status import UserInfoDict, get_user_status_dict, update_user_status
from zerver.models import UserProfile, UserStatus, get_client

def user_status_info(user: UserProfile) -> UserInfoDict:
    if False:
        print('Hello World!')
    user_dict = get_user_status_dict(user.realm_id)
    return user_dict.get(str(user.id), {})

class UserStatusTest(ZulipTestCase):

    def test_basics(self) -> None:
        if False:
            return 10
        hamlet = self.example_user('hamlet')
        client1 = get_client('web')
        client2 = get_client('ZT')
        update_user_status(user_profile_id=hamlet.id, status_text='working', emoji_name=None, emoji_code=None, reaction_type=None, client_id=client1.id)
        self.assertEqual(user_status_info(hamlet), dict(status_text='working'))
        rec_count = UserStatus.objects.filter(user_profile_id=hamlet.id).count()
        self.assertEqual(rec_count, 1)
        update_user_status(user_profile_id=hamlet.id, status_text='out to lunch', emoji_name='car', emoji_code='1f697', reaction_type=UserStatus.UNICODE_EMOJI, client_id=client2.id)
        self.assertEqual(user_status_info(hamlet), dict(status_text='out to lunch', emoji_name='car', emoji_code='1f697', reaction_type=UserStatus.UNICODE_EMOJI))
        rec_count = UserStatus.objects.filter(user_profile_id=hamlet.id).count()
        self.assertEqual(rec_count, 1)
        update_user_status(user_profile_id=hamlet.id, status_text=None, emoji_name=None, emoji_code=None, reaction_type=None, client_id=client2.id)
        self.assertEqual(user_status_info(hamlet), dict(status_text='out to lunch', emoji_name='car', emoji_code='1f697', reaction_type=UserStatus.UNICODE_EMOJI))
        update_user_status(user_profile_id=hamlet.id, status_text='', emoji_name='', emoji_code='', reaction_type=UserStatus.UNICODE_EMOJI, client_id=client2.id)
        self.assertEqual(user_status_info(hamlet), {})
        update_user_status(user_profile_id=hamlet.id, status_text='in a meeting', emoji_name=None, emoji_code=None, reaction_type=None, client_id=client2.id)
        self.assertEqual(user_status_info(hamlet), dict(status_text='in a meeting'))

    def update_status_and_assert_event(self, payload: Dict[str, Any], expected_event: Dict[str, Any], num_events: int=1) -> None:
        if False:
            while True:
                i = 10
        with self.capture_send_event_calls(expected_num_events=num_events) as events:
            result = self.client_post('/json/users/me/status', payload)
        self.assert_json_success(result)
        self.assertEqual(events[0]['event'], expected_event)

    def test_endpoints(self) -> None:
        if False:
            while True:
                i = 10
        hamlet = self.example_user('hamlet')
        realm_id = hamlet.realm_id
        self.login_user(hamlet)
        payload: Dict[str, Any] = {}
        result = self.client_post('/json/users/me/status', payload)
        self.assert_json_error(result, 'Client did not pass any new values.')
        payload = {'status_text': 'In a meeting', 'emoji_code': '1f4bb'}
        result = self.client_post('/json/users/me/status', payload)
        self.assert_json_error(result, 'Client must pass emoji_name if they pass either emoji_code or reaction_type.')
        payload = {'status_text': 'In a meeting', 'emoji_code': '1f4bb', 'emoji_name': 'invalid'}
        result = self.client_post('/json/users/me/status', payload)
        self.assert_json_error(result, "Emoji 'invalid' does not exist")
        payload = {'status_text': 'In a meeting', 'emoji_code': '1f4bb', 'emoji_name': 'car'}
        result = self.client_post('/json/users/me/status', payload)
        self.assert_json_error(result, 'Invalid emoji name.')
        payload = {'status_text': 'In a meeting', 'emoji_code': '1f4bb', 'emoji_name': 'car', 'reaction_type': 'realm_emoji'}
        result = self.client_post('/json/users/me/status', payload)
        self.assert_json_error(result, 'Invalid custom emoji.')
        long_text = 'x' * 61
        payload = dict(status_text=long_text)
        result = self.client_post('/json/users/me/status', payload)
        self.assert_json_error(result, 'status_text is too long (limit: 60 characters)')
        self.update_status_and_assert_event(payload=dict(away=orjson.dumps(True).decode(), status_text='on vacation'), expected_event=dict(type='user_status', user_id=hamlet.id, away=True, status_text='on vacation'), num_events=4)
        self.assertEqual(user_status_info(hamlet), dict(away=True, status_text='on vacation'))
        user = UserProfile.objects.get(id=hamlet.id)
        self.assertEqual(user.presence_enabled, False)
        self.update_status_and_assert_event(payload=dict(emoji_name='car'), expected_event=dict(type='user_status', user_id=hamlet.id, emoji_name='car', emoji_code='1f697', reaction_type=UserStatus.UNICODE_EMOJI))
        self.assertEqual(user_status_info(hamlet), dict(away=True, status_text='on vacation', emoji_name='car', emoji_code='1f697', reaction_type=UserStatus.UNICODE_EMOJI))
        self.update_status_and_assert_event(payload=dict(emoji_name=''), expected_event=dict(type='user_status', user_id=hamlet.id, emoji_name='', emoji_code='', reaction_type=UserStatus.UNICODE_EMOJI))
        self.assertEqual(user_status_info(hamlet), dict(away=True, status_text='on vacation'))
        self.update_status_and_assert_event(payload=dict(away=orjson.dumps(False).decode()), expected_event=dict(type='user_status', user_id=hamlet.id, away=False), num_events=4)
        self.assertEqual(user_status_info(hamlet), dict(status_text='on vacation'))
        user = UserProfile.objects.get(id=hamlet.id)
        self.assertEqual(user.presence_enabled, True)
        self.update_status_and_assert_event(payload=dict(status_text='   in office  '), expected_event=dict(type='user_status', user_id=hamlet.id, status_text='in office'))
        self.assertEqual(user_status_info(hamlet), dict(status_text='in office'))
        self.update_status_and_assert_event(payload=dict(status_text=''), expected_event=dict(type='user_status', user_id=hamlet.id, status_text=''))
        self.assertEqual(get_user_status_dict(realm_id=realm_id), {})
        self.update_status_and_assert_event(payload=dict(away=orjson.dumps(True).decode()), expected_event=dict(type='user_status', user_id=hamlet.id, away=True), num_events=4)
        user = UserProfile.objects.get(id=hamlet.id)
        self.assertEqual(user.presence_enabled, False)
        self.update_status_and_assert_event(payload=dict(status_text='   at the beach  '), expected_event=dict(type='user_status', user_id=hamlet.id, status_text='at the beach'))
        self.assertEqual(user_status_info(hamlet), dict(status_text='at the beach', away=True))