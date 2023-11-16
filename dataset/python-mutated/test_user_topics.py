from datetime import datetime, timezone
from typing import Any, Dict, List
import orjson
import time_machine
from django.utils.timezone import now as timezone_now
from zerver.actions.reactions import check_add_reaction
from zerver.actions.user_settings import do_change_user_setting
from zerver.actions.user_topics import do_set_user_topic_visibility_policy
from zerver.lib.stream_topic import StreamTopicTarget
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import get_subscription
from zerver.lib.user_topics import get_topic_mutes, topic_has_visibility_policy
from zerver.models import UserProfile, UserTopic, get_stream

class MutedTopicsTestsDeprecated(ZulipTestCase):

    def test_get_deactivated_muted_topic(self) -> None:
        if False:
            return 10
        user = self.example_user('hamlet')
        self.login_user(user)
        stream = get_stream('Verona', user.realm)
        mock_date_muted = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        url = '/api/v1/users/me/subscriptions/muted_topics'
        data = {'stream_id': stream.id, 'topic': 'Verona3', 'op': 'add'}
        with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
            result = self.api_patch(user, url, data)
            self.assert_json_success(result)
        stream.deactivated = True
        stream.save()
        self.assertNotIn((stream.name, 'Verona3', mock_date_muted), get_topic_mutes(user))
        self.assertIn((stream.name, 'Verona3', mock_date_muted), get_topic_mutes(user, True))

    def test_user_ids_muting_topic(self) -> None:
        if False:
            while True:
                i = 10
        hamlet = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        realm = hamlet.realm
        stream = get_stream('Verona', realm)
        topic_name = 'teST topic'
        date_muted = datetime(2020, 1, 1, tzinfo=timezone.utc)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.MUTED)
        self.assertEqual(user_ids, set())
        url = '/api/v1/users/me/subscriptions/muted_topics'
        data = {'stream_id': stream.id, 'topic': 'test TOPIC', 'op': 'add'}

        def mute_topic_for_user(user: UserProfile) -> None:
            if False:
                print('Hello World!')
            with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
                result = self.api_patch(user, url, data)
                self.assert_json_success(result)
        mute_topic_for_user(hamlet)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.MUTED)
        self.assertEqual(user_ids, {hamlet.id})
        hamlet_date_muted = UserTopic.objects.filter(user_profile=hamlet, visibility_policy=UserTopic.VisibilityPolicy.MUTED)[0].last_updated
        self.assertEqual(hamlet_date_muted, date_muted)
        mute_topic_for_user(cordelia)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.MUTED)
        self.assertEqual(user_ids, {hamlet.id, cordelia.id})
        cordelia_date_muted = UserTopic.objects.filter(user_profile=cordelia, visibility_policy=UserTopic.VisibilityPolicy.MUTED)[0].last_updated
        self.assertEqual(cordelia_date_muted, date_muted)

    def test_add_muted_topic(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user = self.example_user('hamlet')
        self.login_user(user)
        stream = get_stream('Verona', user.realm)
        url = '/api/v1/users/me/subscriptions/muted_topics'
        payloads: List[Dict[str, object]] = [{'stream': stream.name, 'topic': 'Verona3', 'op': 'add'}, {'stream_id': stream.id, 'topic': 'Verona3', 'op': 'add'}]
        mock_date_muted = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        for data in payloads:
            with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
                result = self.api_patch(user, url, data)
                self.assert_json_success(result)
            self.assertIn((stream.name, 'Verona3', mock_date_muted), get_topic_mutes(user))
            self.assertTrue(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.MUTED))
            do_set_user_topic_visibility_policy(user, stream, 'Verona3', visibility_policy=UserTopic.VisibilityPolicy.INHERIT)
        assert stream.recipient is not None
        result = self.api_patch(user, url, data)
        user_topic_count = UserTopic.objects.count()
        data['topic'] = 'VERONA3'
        with self.assertLogs(level='INFO') as info_logs:
            result = self.api_patch(user, url, data)
            self.assert_json_success(result)
        self.assertEqual(info_logs.output[0], f'INFO:root:User {user.id} tried to set visibility_policy to its current value of {UserTopic.VisibilityPolicy.MUTED}')
        self.assertEqual(UserTopic.objects.count() - user_topic_count, 0)

    def test_remove_muted_topic(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        realm = user.realm
        self.login_user(user)
        stream = get_stream('Verona', realm)
        url = '/api/v1/users/me/subscriptions/muted_topics'
        payloads: List[Dict[str, object]] = [{'stream': stream.name, 'topic': 'vERONA3', 'op': 'remove'}, {'stream_id': stream.id, 'topic': 'vEroNA3', 'op': 'remove'}]
        mock_date_muted = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        for data in payloads:
            do_set_user_topic_visibility_policy(user, stream, 'Verona3', visibility_policy=UserTopic.VisibilityPolicy.MUTED, last_updated=datetime(2020, 1, 1, tzinfo=timezone.utc))
            self.assertIn((stream.name, 'Verona3', mock_date_muted), get_topic_mutes(user))
            result = self.api_patch(user, url, data)
            self.assert_json_success(result)
            self.assertNotIn((stream.name, 'Verona3', mock_date_muted), get_topic_mutes(user))
            self.assertFalse(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.MUTED))

    def test_muted_topic_add_invalid(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        realm = user.realm
        self.login_user(user)
        stream = get_stream('Verona', realm)
        do_set_user_topic_visibility_policy(user, stream, 'Verona3', visibility_policy=UserTopic.VisibilityPolicy.MUTED, last_updated=timezone_now())
        url = '/api/v1/users/me/subscriptions/muted_topics'
        data = {'stream_id': 999999999, 'topic': 'Verona3', 'op': 'add'}
        result = self.api_patch(user, url, data)
        self.assert_json_error(result, 'Invalid stream ID')
        data = {'topic': 'Verona3', 'op': 'add'}
        result = self.api_patch(user, url, data)
        self.assert_json_error(result, "Please supply 'stream'.")
        data = {'stream': stream.name, 'stream_id': stream.id, 'topic': 'Verona3', 'op': 'add'}
        result = self.api_patch(user, url, data)
        self.assert_json_error(result, "Please choose one: 'stream' or 'stream_id'.")

    def test_muted_topic_remove_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user = self.example_user('hamlet')
        realm = user.realm
        self.login_user(user)
        stream = get_stream('Verona', realm)
        url = '/api/v1/users/me/subscriptions/muted_topics'
        data: Dict[str, Any] = {'stream': 'BOGUS', 'topic': 'Verona3', 'op': 'remove'}
        result = self.api_patch(user, url, data)
        self.assert_json_error(result, 'Topic is not muted')
        data = {'stream': stream.name, 'topic': 'BOGUS', 'op': 'remove'}
        with self.assertLogs(level='INFO') as info_logs:
            result = self.api_patch(user, url, data)
            self.assert_json_success(result)
        self.assertEqual(info_logs.output[0], f"INFO:root:User {user.id} tried to remove visibility_policy, which actually doesn't exist")
        data = {'stream_id': 999999999, 'topic': 'BOGUS', 'op': 'remove'}
        result = self.api_patch(user, url, data)
        self.assert_json_error(result, 'Topic is not muted')
        data = {'topic': 'Verona3', 'op': 'remove'}
        result = self.api_patch(user, url, data)
        self.assert_json_error(result, "Please supply 'stream'.")
        data = {'stream': stream.name, 'stream_id': stream.id, 'topic': 'Verona3', 'op': 'remove'}
        result = self.api_patch(user, url, data)
        self.assert_json_error(result, "Please choose one: 'stream' or 'stream_id'.")

class MutedTopicsTests(ZulipTestCase):

    def test_get_deactivated_muted_topic(self) -> None:
        if False:
            print('Hello World!')
        user = self.example_user('hamlet')
        self.login_user(user)
        stream = get_stream('Verona', user.realm)
        url = '/api/v1/user_topics'
        data = {'stream_id': stream.id, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.MUTED}
        mock_date_muted = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
            result = self.api_post(user, url, data)
            self.assert_json_success(result)
        stream.deactivated = True
        stream.save()
        self.assertNotIn((stream.name, 'Verona3', mock_date_muted), get_topic_mutes(user))
        self.assertIn((stream.name, 'Verona3', mock_date_muted), get_topic_mutes(user, True))

    def test_user_ids_muting_topic(self) -> None:
        if False:
            print('Hello World!')
        hamlet = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        realm = hamlet.realm
        stream = get_stream('Verona', realm)
        topic_name = 'teST topic'
        date_muted = datetime(2020, 1, 1, tzinfo=timezone.utc)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.MUTED)
        self.assertEqual(user_ids, set())
        url = '/api/v1/user_topics'

        def set_topic_visibility_for_user(user: UserProfile, visibility_policy: int) -> None:
            if False:
                return 10
            data = {'stream_id': stream.id, 'topic': 'test TOPIC', 'visibility_policy': visibility_policy}
            with time_machine.travel(date_muted, tick=False):
                result = self.api_post(user, url, data)
                self.assert_json_success(result)
        set_topic_visibility_for_user(hamlet, UserTopic.VisibilityPolicy.MUTED)
        set_topic_visibility_for_user(cordelia, UserTopic.VisibilityPolicy.UNMUTED)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.MUTED)
        self.assertEqual(user_ids, {hamlet.id})
        hamlet_date_muted = UserTopic.objects.filter(user_profile=hamlet, visibility_policy=UserTopic.VisibilityPolicy.MUTED)[0].last_updated
        self.assertEqual(hamlet_date_muted, date_muted)
        set_topic_visibility_for_user(cordelia, UserTopic.VisibilityPolicy.MUTED)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.MUTED)
        self.assertEqual(user_ids, {hamlet.id, cordelia.id})
        cordelia_date_muted = UserTopic.objects.filter(user_profile=cordelia, visibility_policy=UserTopic.VisibilityPolicy.MUTED)[0].last_updated
        self.assertEqual(cordelia_date_muted, date_muted)

    def test_add_muted_topic(self) -> None:
        if False:
            print('Hello World!')
        user = self.example_user('hamlet')
        self.login_user(user)
        stream = get_stream('Verona', user.realm)
        url = '/api/v1/user_topics'
        data = {'stream_id': stream.id, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.MUTED}
        mock_date_muted = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        with self.capture_send_event_calls(expected_num_events=2) as events:
            with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
                result = self.api_post(user, url, data)
                self.assert_json_success(result)
        self.assertTrue(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.MUTED))
        user_topic_event: Dict[str, Any] = {'type': 'user_topic', 'stream_id': stream.id, 'topic_name': 'Verona3', 'last_updated': mock_date_muted, 'visibility_policy': UserTopic.VisibilityPolicy.MUTED}
        muted_topics_event = dict(type='muted_topics', muted_topics=get_topic_mutes(user))
        self.assertEqual(events[0]['event'], muted_topics_event)
        self.assertEqual(events[1]['event'], user_topic_event)
        user_topic_count = UserTopic.objects.count()
        data['topic'] = 'VERONA3'
        with self.assertLogs(level='INFO') as info_logs:
            result = self.api_post(user, url, data)
            self.assert_json_success(result)
        self.assertEqual(info_logs.output[0], f'INFO:root:User {user.id} tried to set visibility_policy to its current value of {UserTopic.VisibilityPolicy.MUTED}')
        self.assertEqual(UserTopic.objects.count() - user_topic_count, 0)

    def test_remove_muted_topic(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        realm = user.realm
        self.login_user(user)
        stream = get_stream('Verona', realm)
        do_set_user_topic_visibility_policy(user, stream, 'Verona3', visibility_policy=UserTopic.VisibilityPolicy.MUTED, last_updated=datetime(2020, 1, 1, tzinfo=timezone.utc))
        self.assertTrue(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.MUTED))
        url = '/api/v1/user_topics'
        data = {'stream_id': stream.id, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.INHERIT}
        mock_date_mute_removed = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        with self.capture_send_event_calls(expected_num_events=2) as events:
            with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
                result = self.api_post(user, url, data)
                self.assert_json_success(result)
        self.assertFalse(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.MUTED))
        user_topic_event: Dict[str, Any] = {'type': 'user_topic', 'stream_id': stream.id, 'topic_name': data['topic'], 'last_updated': mock_date_mute_removed, 'visibility_policy': UserTopic.VisibilityPolicy.INHERIT}
        muted_topics_event = dict(type='muted_topics', muted_topics=get_topic_mutes(user))
        self.assertEqual(events[0]['event'], muted_topics_event)
        self.assertEqual(events[1]['event'], user_topic_event)
        with self.assertLogs(level='INFO') as info_logs:
            result = self.api_post(user, url, data)
            self.assert_json_success(result)
        self.assertEqual(info_logs.output[0], f"INFO:root:User {user.id} tried to remove visibility_policy, which actually doesn't exist")

    def test_muted_topic_add_invalid(self) -> None:
        if False:
            while True:
                i = 10
        user = self.example_user('hamlet')
        self.login_user(user)
        url = '/api/v1/user_topics'
        data = {'stream_id': 999999999, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.MUTED}
        result = self.api_post(user, url, data)
        self.assert_json_error(result, 'Invalid stream ID')

    def test_muted_topic_remove_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user = self.example_user('hamlet')
        self.login_user(user)
        url = '/api/v1/user_topics'
        data = {'stream_id': 999999999, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.INHERIT}
        result = self.api_post(user, url, data)
        self.assert_json_error(result, 'Invalid stream ID')

class UnmutedTopicsTests(ZulipTestCase):

    def test_user_ids_unmuting_topic(self) -> None:
        if False:
            print('Hello World!')
        hamlet = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        realm = hamlet.realm
        stream = get_stream('Verona', realm)
        topic_name = 'teST topic'
        date_unmuted = datetime(2020, 1, 1, tzinfo=timezone.utc)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        url = '/api/v1/user_topics'

        def set_topic_visibility_for_user(user: UserProfile, visibility_policy: int) -> None:
            if False:
                for i in range(10):
                    print('nop')
            data = {'stream_id': stream.id, 'topic': 'test TOPIC', 'visibility_policy': visibility_policy}
            with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
                result = self.api_post(user, url, data)
                self.assert_json_success(result)
        set_topic_visibility_for_user(hamlet, UserTopic.VisibilityPolicy.UNMUTED)
        set_topic_visibility_for_user(cordelia, UserTopic.VisibilityPolicy.MUTED)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id})
        hamlet_date_unmuted = UserTopic.objects.filter(user_profile=hamlet, visibility_policy=UserTopic.VisibilityPolicy.UNMUTED)[0].last_updated
        self.assertEqual(hamlet_date_unmuted, date_unmuted)
        set_topic_visibility_for_user(cordelia, UserTopic.VisibilityPolicy.UNMUTED)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id, cordelia.id})
        cordelia_date_unmuted = UserTopic.objects.filter(user_profile=cordelia, visibility_policy=UserTopic.VisibilityPolicy.UNMUTED)[0].last_updated
        self.assertEqual(cordelia_date_unmuted, date_unmuted)

    def test_add_unmuted_topic(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        self.login_user(user)
        stream = get_stream('Verona', user.realm)
        url = '/api/v1/user_topics'
        data = {'stream_id': stream.id, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.UNMUTED}
        mock_date_unmuted = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        with self.capture_send_event_calls(expected_num_events=2) as events:
            with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
                result = self.api_post(user, url, data)
                self.assert_json_success(result)
        self.assertTrue(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.UNMUTED))
        user_topic_event: Dict[str, Any] = {'type': 'user_topic', 'stream_id': stream.id, 'topic_name': 'Verona3', 'last_updated': mock_date_unmuted, 'visibility_policy': UserTopic.VisibilityPolicy.UNMUTED}
        muted_topics_event = dict(type='muted_topics', muted_topics=get_topic_mutes(user))
        self.assertEqual(events[0]['event'], muted_topics_event)
        self.assertEqual(events[1]['event'], user_topic_event)
        user_topic_count = UserTopic.objects.count()
        data['topic'] = 'VERONA3'
        with self.assertLogs(level='INFO') as info_logs:
            result = self.api_post(user, url, data)
            self.assert_json_success(result)
        self.assertEqual(info_logs.output[0], f'INFO:root:User {user.id} tried to set visibility_policy to its current value of {UserTopic.VisibilityPolicy.UNMUTED}')
        self.assertEqual(UserTopic.objects.count() - user_topic_count, 0)

    def test_remove_unmuted_topic(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user = self.example_user('hamlet')
        realm = user.realm
        self.login_user(user)
        stream = get_stream('Verona', realm)
        do_set_user_topic_visibility_policy(user, stream, 'Verona3', visibility_policy=UserTopic.VisibilityPolicy.UNMUTED, last_updated=datetime(2020, 1, 1, tzinfo=timezone.utc))
        self.assertTrue(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.UNMUTED))
        url = '/api/v1/user_topics'
        data = {'stream_id': stream.id, 'topic': 'vEroNA3', 'visibility_policy': UserTopic.VisibilityPolicy.INHERIT}
        mock_date_unmute_removed = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        with self.capture_send_event_calls(expected_num_events=2) as events:
            with time_machine.travel(datetime(2020, 1, 1, tzinfo=timezone.utc), tick=False):
                result = self.api_post(user, url, data)
                self.assert_json_success(result)
        self.assertFalse(topic_has_visibility_policy(user, stream.id, 'verona3', UserTopic.VisibilityPolicy.UNMUTED))
        user_topic_event: Dict[str, Any] = {'type': 'user_topic', 'stream_id': stream.id, 'topic_name': data['topic'], 'last_updated': mock_date_unmute_removed, 'visibility_policy': UserTopic.VisibilityPolicy.INHERIT}
        muted_topics_event = dict(type='muted_topics', muted_topics=get_topic_mutes(user))
        self.assertEqual(events[0]['event'], muted_topics_event)
        self.assertEqual(events[1]['event'], user_topic_event)
        with self.assertLogs(level='INFO') as info_logs:
            result = self.api_post(user, url, data)
            self.assert_json_success(result)
        self.assertEqual(info_logs.output[0], f"INFO:root:User {user.id} tried to remove visibility_policy, which actually doesn't exist")

    def test_unmuted_topic_add_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user = self.example_user('hamlet')
        self.login_user(user)
        url = '/api/v1/user_topics'
        data = {'stream_id': 999999999, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.UNMUTED}
        result = self.api_post(user, url, data)
        self.assert_json_error(result, 'Invalid stream ID')

    def test_unmuted_topic_remove_invalid(self) -> None:
        if False:
            print('Hello World!')
        user = self.example_user('hamlet')
        self.login_user(user)
        url = '/api/v1/user_topics'
        data = {'stream_id': 999999999, 'topic': 'Verona3', 'visibility_policy': UserTopic.VisibilityPolicy.INHERIT}
        result = self.api_post(user, url, data)
        self.assert_json_error(result, 'Invalid stream ID')

class AutomaticallyFollowTopicsTests(ZulipTestCase):

    def test_automatically_follow_topic_on_initiation(self) -> None:
        if False:
            while True:
                i = 10
        hamlet = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        iago = self.example_user('iago')
        stream = get_stream('Verona', hamlet.realm)
        topic_name = 'teST topic'
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, set())
        for user in [hamlet, cordelia]:
            do_change_user_setting(user, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {hamlet.id})
        self.send_stream_message(cordelia, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {hamlet.id})
        do_change_user_setting(iago, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_NEVER, acting_user=None)
        self.send_stream_message(iago, stream_name=stream.name, topic_name='New Topic')
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name='New Topic')
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, set())
        private_stream = self.make_stream(stream_name='private stream', invite_only=True)
        self.subscribe(iago, private_stream.name)
        self.send_stream_message(iago, private_stream.name)
        self.subscribe(hamlet, private_stream.name)
        self.send_stream_message(hamlet, private_stream.name)
        stream_topic_target = StreamTopicTarget(stream_id=private_stream.id, topic_name='test')
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {hamlet.id})

    def test_automatically_follow_topic_on_send(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', hamlet.realm)
        topic_name = 'teST topic'
        self.send_stream_message(aaron, stream.name, 'hello', topic_name)
        do_change_user_setting(hamlet, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND, acting_user=None)
        do_change_user_setting(aaron, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, set())
        self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        self.send_stream_message(aaron, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {hamlet.id})

    def test_automatically_follow_topic_on_participation_send_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', hamlet.realm)
        topic_name = 'teST topic'
        do_change_user_setting(hamlet, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, set())
        self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        self.send_stream_message(aaron, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {hamlet.id})

    def test_automatically_follow_topic_on_participation_add_reaction(self) -> None:
        if False:
            i = 10
            return i + 15
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        do_change_user_setting(cordelia, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, set())
        message_id = self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        check_add_reaction(user_profile=cordelia, message_id=message_id, emoji_name='smile', emoji_code=None, reaction_type=None)
        check_add_reaction(user_profile=aaron, message_id=message_id, emoji_name='smile', emoji_code=None, reaction_type=None)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {cordelia.id})
        sub = get_subscription(stream.name, cordelia)
        sub.is_muted = True
        sub.save()
        do_change_user_setting(cordelia, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_NEVER, acting_user=None)
        do_change_user_setting(cordelia, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        check_add_reaction(user_profile=cordelia, message_id=message_id, emoji_name='plus', emoji_code=None, reaction_type=None)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {cordelia.id})
        do_set_user_topic_visibility_policy(cordelia, stream, topic_name, visibility_policy=UserTopic.VisibilityPolicy.MUTED)
        check_add_reaction(user_profile=cordelia, message_id=message_id, emoji_name='heart', emoji_code=None, reaction_type=None)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {cordelia.id})
        check_add_reaction(user_profile=cordelia, message_id=message_id, emoji_name='tada', emoji_code=None, reaction_type=None)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {cordelia.id})

    def test_automatically_follow_topic_on_participation_participate_in_poll(self) -> None:
        if False:
            print('Hello World!')
        iago = self.example_user('iago')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        do_change_user_setting(iago, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, set())
        payload = dict(type='stream', to=orjson.dumps(stream.name).decode(), topic=topic_name, content='/poll Preference?\n\nyes\nno')
        result = self.api_post(hamlet, '/api/v1/messages', payload)
        self.assert_json_success(result)
        message = self.get_last_message()

        def participate_in_poll(user: UserProfile, data: Dict[str, object]) -> None:
            if False:
                return 10
            content = orjson.dumps(data).decode()
            payload = dict(message_id=message.id, msg_type='widget', content=content)
            result = self.api_post(user, '/api/v1/submessage', payload)
            self.assert_json_success(result)
        participate_in_poll(iago, dict(type='vote', key='1,1', vote=1))
        participate_in_poll(aaron, dict(type='new_option', idx=7, option='maybe'))
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {iago.id})

    def test_automatically_follow_topic_on_participation_edit_todo_list(self) -> None:
        if False:
            return 10
        othello = self.example_user('othello')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        do_change_user_setting(othello, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_follow_topics_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, set())
        payload = dict(type='stream', to=orjson.dumps(stream.name).decode(), topic=topic_name, content='/todo')
        result = self.api_post(hamlet, '/api/v1/messages', payload)
        self.assert_json_success(result)
        message = self.get_last_message()

        def edit_todo_list(user: UserProfile, data: Dict[str, object]) -> None:
            if False:
                return 10
            content = orjson.dumps(data).decode()
            payload = dict(message_id=message.id, msg_type='widget', content=content)
            result = self.api_post(user, '/api/v1/submessage', payload)
            self.assert_json_success(result)
        edit_todo_list(othello, dict(type='new_task', key=7, task='eat', desc='', completed=False))
        edit_todo_list(aaron, dict(type='strike', key='5,9'))
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.FOLLOWED)
        self.assertEqual(user_ids, {othello.id})

class AutomaticallyUnmuteTopicsTests(ZulipTestCase):

    def test_automatically_unmute_topic_on_initiation(self) -> None:
        if False:
            print('Hello World!')
        hamlet = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        iago = self.example_user('iago')
        stream = get_stream('Verona', hamlet.realm)
        topic_name = 'teST topic'
        for user in [hamlet, cordelia, iago]:
            sub = get_subscription(stream.name, user)
            sub.is_muted = True
            sub.save()
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        for user in [hamlet, cordelia]:
            do_change_user_setting(user, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id})
        self.send_stream_message(cordelia, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id})
        do_change_user_setting(iago, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_NEVER, acting_user=None)
        self.send_stream_message(iago, stream_name=stream.name, topic_name='New Topic')
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name='New Topic')
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        private_stream = self.make_stream(stream_name='private stream', invite_only=True)
        self.subscribe(iago, private_stream.name)
        self.send_stream_message(iago, private_stream.name)
        self.subscribe(hamlet, private_stream.name)
        sub = get_subscription(private_stream.name, hamlet)
        sub.is_muted = True
        sub.save()
        self.send_stream_message(hamlet, private_stream.name)
        stream_topic_target = StreamTopicTarget(stream_id=private_stream.id, topic_name='test')
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id})

    def test_automatically_unmute_topic_on_send(self) -> None:
        if False:
            while True:
                i = 10
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', hamlet.realm)
        topic_name = 'teST topic'
        self.send_stream_message(aaron, stream.name, 'hello', topic_name)
        for user in [hamlet, aaron]:
            sub = get_subscription(stream.name, user)
            sub.is_muted = True
            sub.save()
        do_change_user_setting(hamlet, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND, acting_user=None)
        do_change_user_setting(aaron, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        self.send_stream_message(aaron, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id})

    def test_automatically_unmute_topic_on_participation_send_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', hamlet.realm)
        topic_name = 'teST topic'
        for user in [hamlet, aaron]:
            sub = get_subscription(stream.name, user)
            sub.is_muted = True
            sub.save()
        do_change_user_setting(hamlet, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        self.send_stream_message(aaron, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id})

    def test_automatically_unmute_topic_on_participation_add_reaction(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        for user in [cordelia, hamlet, aaron]:
            sub = get_subscription(stream.name, user)
            sub.is_muted = True
            sub.save()
        do_change_user_setting(cordelia, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        message_id = self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        check_add_reaction(user_profile=cordelia, message_id=message_id, emoji_name='smile', emoji_code=None, reaction_type=None)
        check_add_reaction(user_profile=aaron, message_id=message_id, emoji_name='smile', emoji_code=None, reaction_type=None)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {cordelia.id})

    def test_automatically_unmute_topic_on_participation_participate_in_poll(self) -> None:
        if False:
            i = 10
            return i + 15
        iago = self.example_user('iago')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        for user in [iago, hamlet, aaron]:
            sub = get_subscription(stream.name, user)
            sub.is_muted = True
            sub.save()
        do_change_user_setting(iago, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        payload = dict(type='stream', to=orjson.dumps(stream.name).decode(), topic=topic_name, content='/poll Preference?\n\nyes\nno')
        result = self.api_post(hamlet, '/api/v1/messages', payload)
        self.assert_json_success(result)
        message = self.get_last_message()

        def participate_in_poll(user: UserProfile, data: Dict[str, object]) -> None:
            if False:
                i = 10
                return i + 15
            content = orjson.dumps(data).decode()
            payload = dict(message_id=message.id, msg_type='widget', content=content)
            result = self.api_post(user, '/api/v1/submessage', payload)
            self.assert_json_success(result)
        participate_in_poll(iago, dict(type='vote', key='1,1', vote=1))
        participate_in_poll(aaron, dict(type='new_option', idx=7, option='maybe'))
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {iago.id})

    def test_automatically_unmute_topic_on_participation_edit_todo_list(self) -> None:
        if False:
            while True:
                i = 10
        othello = self.example_user('othello')
        hamlet = self.example_user('hamlet')
        aaron = self.example_user('aaron')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        for user in [othello, hamlet, aaron]:
            sub = get_subscription(stream.name, user)
            sub.is_muted = True
            sub.save()
        do_change_user_setting(othello, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        do_change_user_setting(aaron, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        payload = dict(type='stream', to=orjson.dumps(stream.name).decode(), topic=topic_name, content='/todo')
        result = self.api_post(hamlet, '/api/v1/messages', payload)
        self.assert_json_success(result)
        message = self.get_last_message()

        def edit_todo_list(user: UserProfile, data: Dict[str, object]) -> None:
            if False:
                i = 10
                return i + 15
            content = orjson.dumps(data).decode()
            payload = dict(message_id=message.id, msg_type='widget', content=content)
            result = self.api_post(user, '/api/v1/submessage', payload)
            self.assert_json_success(result)
        edit_todo_list(othello, dict(type='new_task', key=7, task='eat', desc='', completed=False))
        edit_todo_list(aaron, dict(type='strike', key='5,9'))
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {othello.id})

    def test_only_automatically_increase_visibility_policy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        aaron = self.example_user('aaron')
        hamlet = self.example_user('hamlet')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        for user in [hamlet, aaron]:
            sub = get_subscription(stream.name, user)
            sub.is_muted = True
            sub.save()
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        do_set_user_topic_visibility_policy(aaron, stream, topic_name, visibility_policy=UserTopic.VisibilityPolicy.FOLLOWED)
        do_change_user_setting(aaron, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        self.send_stream_message(aaron, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        topic_name = 'new Topic'
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        do_set_user_topic_visibility_policy(hamlet, stream, topic_name, visibility_policy=UserTopic.VisibilityPolicy.MUTED)
        do_change_user_setting(hamlet, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        self.send_stream_message(hamlet, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, {hamlet.id})

    def test_automatically_unmute_policy_unmuted_stream(self) -> None:
        if False:
            i = 10
            return i + 15
        aaron = self.example_user('aaron')
        cordelia = self.example_user('cordelia')
        stream = get_stream('Verona', aaron.realm)
        topic_name = 'teST topic'
        stream_topic_target = StreamTopicTarget(stream_id=stream.id, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        do_change_user_setting(aaron, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, acting_user=None)
        self.send_stream_message(aaron, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())
        do_set_user_topic_visibility_policy(cordelia, stream, topic_name, visibility_policy=UserTopic.VisibilityPolicy.MUTED)
        do_change_user_setting(cordelia, 'automatically_unmute_topics_in_muted_streams_policy', UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION, acting_user=None)
        self.send_stream_message(cordelia, stream_name=stream.name, topic_name=topic_name)
        user_ids = stream_topic_target.user_ids_with_visibility_policy(UserTopic.VisibilityPolicy.UNMUTED)
        self.assertEqual(user_ids, set())