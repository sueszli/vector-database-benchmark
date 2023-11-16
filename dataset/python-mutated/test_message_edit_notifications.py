from typing import Any, Dict, Mapping, Union
from unittest import mock
from django.utils.timezone import now as timezone_now
from zerver.actions.user_settings import do_change_user_setting
from zerver.actions.user_topics import do_set_user_topic_visibility_policy
from zerver.lib.push_notifications import get_apns_badge_count, get_apns_badge_count_future
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import mock_queue_publish
from zerver.models import NotificationTriggers, Subscription, UserPresence, UserTopic, get_stream
from zerver.tornado.event_queue import maybe_enqueue_notifications

class EditMessageSideEffectsTest(ZulipTestCase):

    def _assert_update_does_not_notify_anybody(self, message_id: int, content: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        url = '/json/messages/' + str(message_id)
        request = dict(content=content)
        with mock.patch('zerver.tornado.event_queue.maybe_enqueue_notifications') as m:
            result = self.client_patch(url, request)
        self.assert_json_success(result)
        self.assertFalse(m.called)

    def test_updates_with_pm_mention(self) -> None:
        if False:
            print('Hello World!')
        hamlet = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        self.login_user(hamlet)
        message_id = self.send_personal_message(hamlet, cordelia, content='no mention')
        self._assert_update_does_not_notify_anybody(message_id=message_id, content="now we mention @**Cordelia, Lear's daughter**")

    def _login_and_send_original_stream_message(self, content: str, enable_online_push_notifications: bool=False) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Note our conventions here:\n\n            Hamlet is our logged in user (and sender).\n            Cordelia is the receiver we care about.\n            Scotland is the stream we send messages to.\n        '
        hamlet = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        cordelia.enable_online_push_notifications = enable_online_push_notifications
        cordelia.save()
        self.login_user(hamlet)
        self.subscribe(hamlet, 'Scotland')
        self.subscribe(cordelia, 'Scotland')
        message_id = self.send_stream_message(hamlet, 'Scotland', content=content)
        return message_id

    def _get_queued_data_for_message_update(self, message_id: int, content: str, expect_short_circuit: bool=False) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        This function updates a message with a post to\n        /json/messages/(message_id).\n\n        By using mocks, we are able to capture two pieces of data:\n\n            enqueue_kwargs: These are the arguments passed in to\n                            maybe_enqueue_notifications.\n\n            queue_messages: These are the messages that\n                            maybe_enqueue_notifications actually\n                            puts on the queue.\n\n        Using this helper allows you to construct a test that goes\n        pretty deep into the missed-messages codepath, without actually\n        queuing the final messages.\n        '
        url = '/json/messages/' + str(message_id)
        request = dict(content=content)
        with mock.patch('zerver.tornado.event_queue.maybe_enqueue_notifications') as m:
            result = self.client_patch(url, request)
        cordelia = self.example_user('cordelia')
        cordelia_calls = [call_args for call_args in m.call_args_list if call_args[1]['user_notifications_data'].user_id == cordelia.id]
        if expect_short_circuit:
            self.assert_length(cordelia_calls, 0)
            return {}
        self.assert_length(cordelia_calls, 1)
        enqueue_kwargs = cordelia_calls[0][1]
        queue_messages = []

        def fake_publish(queue_name: str, event: Union[Mapping[str, Any], str], *args: Any) -> None:
            if False:
                i = 10
                return i + 15
            queue_messages.append(dict(queue_name=queue_name, event=event))
        with mock_queue_publish('zerver.tornado.event_queue.queue_json_publish', side_effect=fake_publish) as m:
            maybe_enqueue_notifications(**enqueue_kwargs)
        self.assert_json_success(result)
        return dict(enqueue_kwargs=enqueue_kwargs, queue_messages=queue_messages)

    def _send_and_update_message(self, original_content: str, updated_content: str, enable_online_push_notifications: bool=False, expect_short_circuit: bool=False, connected_to_zulip: bool=False, present_on_web: bool=False) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        message_id = self._login_and_send_original_stream_message(content=original_content, enable_online_push_notifications=enable_online_push_notifications)
        if present_on_web:
            self._make_cordelia_present_on_web()
        if connected_to_zulip:
            with self._cordelia_connected_to_zulip():
                info = self._get_queued_data_for_message_update(message_id=message_id, content=updated_content, expect_short_circuit=expect_short_circuit)
        else:
            info = self._get_queued_data_for_message_update(message_id=message_id, content=updated_content, expect_short_circuit=expect_short_circuit)
        return dict(message_id=message_id, info=info)

    def test_updates_with_stream_mention(self) -> None:
        if False:
            i = 10
            return i + 15
        original_content = 'no mention'
        updated_content = "now we mention @**Cordelia, Lear's daughter**"
        notification_message_data = self._send_and_update_message(original_content, updated_content)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, mention_email_notify=True, mention_push_notify=True, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        queue_messages = info['queue_messages']
        self.assert_length(queue_messages, 2)
        self.assertEqual(queue_messages[0]['queue_name'], 'missedmessage_mobile_notifications')
        mobile_event = queue_messages[0]['event']
        self.assertEqual(mobile_event['user_profile_id'], cordelia.id)
        self.assertEqual(mobile_event['trigger'], NotificationTriggers.MENTION)
        self.assertEqual(queue_messages[1]['queue_name'], 'missedmessage_emails')
        email_event = queue_messages[1]['event']
        self.assertEqual(email_event['user_profile_id'], cordelia.id)
        self.assertEqual(email_event['trigger'], NotificationTriggers.MENTION)

    def test_second_mention_is_ignored(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        original_content = "hello @**Cordelia, Lear's daughter**"
        updated_content = "re-mention @**Cordelia, Lear's daughter**"
        self._send_and_update_message(original_content, updated_content, expect_short_circuit=True)

    def _turn_on_stream_push_for_cordelia(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        conventions:\n            Cordelia is the message receiver we care about.\n            Scotland is our stream.\n        '
        cordelia = self.example_user('cordelia')
        stream = self.subscribe(cordelia, 'Scotland')
        recipient = stream.recipient
        cordelia_subscription = Subscription.objects.get(user_profile_id=cordelia.id, recipient=recipient)
        cordelia_subscription.push_notifications = True
        cordelia_subscription.save()

    def test_updates_with_stream_push_notify(self) -> None:
        if False:
            return 10
        self._turn_on_stream_push_for_cordelia()
        original_content = 'no mention'
        updated_content = 'nothing special about updated message'
        self._send_and_update_message(original_content, updated_content, expect_short_circuit=True)

    def _cordelia_connected_to_zulip(self) -> Any:
        if False:
            return 10
        "\n        Right now the easiest way to make Cordelia look\n        connected to Zulip is to mock the function below.\n\n        This is a bit blunt, as it affects other users too,\n        but we only really look at Cordelia's data, anyway.\n        "
        return mock.patch('zerver.tornado.event_queue.receiver_is_off_zulip', return_value=False)

    def test_stream_push_notify_for_sorta_present_user(self) -> None:
        if False:
            i = 10
            return i + 15
        self._turn_on_stream_push_for_cordelia()
        original_content = 'no mention'
        updated_content = 'nothing special about updated message'
        self._send_and_update_message(original_content, updated_content, expect_short_circuit=True, connected_to_zulip=True)

    def _make_cordelia_present_on_web(self) -> None:
        if False:
            print('Hello World!')
        cordelia = self.example_user('cordelia')
        now = timezone_now()
        UserPresence.objects.create(user_profile_id=cordelia.id, realm_id=cordelia.realm_id, last_connected_time=now, last_active_time=now)

    def test_stream_push_notify_for_fully_present_user(self) -> None:
        if False:
            i = 10
            return i + 15
        self._turn_on_stream_push_for_cordelia()
        original_content = 'no mention'
        updated_content = 'nothing special about updated message'
        self._send_and_update_message(original_content, updated_content, expect_short_circuit=True, connected_to_zulip=True, present_on_web=True)

    def test_online_push_enabled_for_fully_present_mentioned_user(self) -> None:
        if False:
            return 10
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        original_content = 'no mention'
        updated_content = "newly mention @**Cordelia, Lear's daughter**"
        notification_message_data = self._send_and_update_message(original_content, updated_content, enable_online_push_notifications=True, connected_to_zulip=True, present_on_web=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, mention_push_notify=True, mention_email_notify=True, online_push_enabled=True, idle=False, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        queue_messages = info['queue_messages']
        self.assert_length(queue_messages, 1)

    def test_online_push_enabled_for_fully_present_boring_user(self) -> None:
        if False:
            return 10
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        original_content = 'no mention'
        updated_content = 'nothing special about updated message'
        notification_message_data = self._send_and_update_message(original_content, updated_content, enable_online_push_notifications=True, connected_to_zulip=True, present_on_web=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, online_push_enabled=True, idle=False, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        queue_messages = info['queue_messages']
        self.assert_length(queue_messages, 0)

    def test_updates_with_stream_mention_of_sorta_present_user(self) -> None:
        if False:
            while True:
                i = 10
        cordelia = self.example_user('cordelia')
        original_content = 'no mention'
        updated_content = "now we mention @**Cordelia, Lear's daughter**"
        notification_message_data = self._send_and_update_message(original_content, updated_content, connected_to_zulip=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, message_id=message_id, acting_user_id=self.example_user('hamlet').id, mention_email_notify=True, mention_push_notify=True, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        self.assert_length(info['queue_messages'], 2)

    def test_updates_with_topic_wildcard_mention_in_followed_topic(self) -> None:
        if False:
            print('Hello World!')
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        self.subscribe(cordelia, 'Scotland')
        do_change_user_setting(cordelia, 'enable_followed_topic_email_notifications', False, acting_user=None)
        do_change_user_setting(cordelia, 'enable_followed_topic_push_notifications', False, acting_user=None)
        do_change_user_setting(cordelia, 'wildcard_mentions_notify', False, acting_user=None)
        do_set_user_topic_visibility_policy(user_profile=cordelia, stream=get_stream('Scotland', cordelia.realm), topic='test', visibility_policy=UserTopic.VisibilityPolicy.FOLLOWED)
        self.send_stream_message(cordelia, 'Scotland')
        original_content = 'no mention'
        updated_content = 'now we mention @**topic**'
        notification_message_data = self._send_and_update_message(original_content, updated_content, connected_to_zulip=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, topic_wildcard_mention_in_followed_topic_email_notify=True, topic_wildcard_mention_in_followed_topic_push_notify=True, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        self.assert_length(info['queue_messages'], 2)

    def test_updates_with_stream_wildcard_mention_in_followed_topic(self) -> None:
        if False:
            return 10
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        self.subscribe(cordelia, 'Scotland')
        do_change_user_setting(cordelia, 'enable_followed_topic_email_notifications', False, acting_user=None)
        do_change_user_setting(cordelia, 'enable_followed_topic_push_notifications', False, acting_user=None)
        do_change_user_setting(cordelia, 'wildcard_mentions_notify', False, acting_user=None)
        do_set_user_topic_visibility_policy(user_profile=cordelia, stream=get_stream('Scotland', cordelia.realm), topic='test', visibility_policy=UserTopic.VisibilityPolicy.FOLLOWED)
        original_content = 'no mention'
        updated_content = 'now we mention @**all**'
        notification_message_data = self._send_and_update_message(original_content, updated_content, connected_to_zulip=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, stream_wildcard_mention_in_followed_topic_email_notify=True, stream_wildcard_mention_in_followed_topic_push_notify=True, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        self.assert_length(info['queue_messages'], 2)

    def test_updates_with_topic_wildcard_mention(self) -> None:
        if False:
            return 10
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        self.subscribe(cordelia, 'Scotland')
        self.send_stream_message(cordelia, 'Scotland')
        original_content = 'no mention'
        updated_content = 'now we mention @**topic**'
        notification_message_data = self._send_and_update_message(original_content, updated_content, connected_to_zulip=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, topic_wildcard_mention_email_notify=True, topic_wildcard_mention_push_notify=True, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        self.assert_length(info['queue_messages'], 2)

    def test_updates_with_stream_wildcard_mention(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        original_content = 'no mention'
        updated_content = 'now we mention @**all**'
        notification_message_data = self._send_and_update_message(original_content, updated_content, connected_to_zulip=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, stream_wildcard_mention_email_notify=True, stream_wildcard_mention_push_notify=True, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        self.assert_length(info['queue_messages'], 2)

    def test_updates_with_upgrade_wildcard_mention(self) -> None:
        if False:
            while True:
                i = 10
        original_content = 'Mention @**all**'
        updated_content = "now we mention @**Cordelia, Lear's daughter**"
        self._send_and_update_message(original_content, updated_content, expect_short_circuit=True, connected_to_zulip=True)

    def test_updates_with_upgrade_wildcard_mention_disabled(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cordelia = self.example_user('cordelia')
        cordelia.wildcard_mentions_notify = False
        cordelia.save()
        original_content = 'Mention @**all**'
        updated_content = "now we mention @**Cordelia, Lear's daughter**"
        self._send_and_update_message(original_content, updated_content, expect_short_circuit=True, connected_to_zulip=True)

    def test_updates_with_stream_mention_of_fully_present_user(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cordelia = self.example_user('cordelia')
        hamlet = self.example_user('hamlet')
        original_content = 'no mention'
        updated_content = "now we mention @**Cordelia, Lear's daughter**"
        notification_message_data = self._send_and_update_message(original_content, updated_content, connected_to_zulip=True, present_on_web=True)
        message_id = notification_message_data['message_id']
        info = notification_message_data['info']
        expected_enqueue_kwargs = self.get_maybe_enqueue_notifications_parameters(user_id=cordelia.id, acting_user_id=hamlet.id, message_id=message_id, mention_email_notify=True, mention_push_notify=True, idle=False, already_notified={})
        self.assertEqual(info['enqueue_kwargs'], expected_enqueue_kwargs)
        self.assert_length(info['queue_messages'], 0)

    @mock.patch('zerver.lib.push_notifications.push_notifications_enabled', return_value=True)
    def test_clear_notification_when_mention_removed(self, mock_push_notifications: mock.MagicMock) -> None:
        if False:
            return 10
        mentioned_user = self.example_user('iago')
        self.assertEqual(get_apns_badge_count(mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(mentioned_user), 0)
        message_id = self._login_and_send_original_stream_message(content='@**Iago**')
        self.assertEqual(get_apns_badge_count(mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(mentioned_user), 1)
        self._get_queued_data_for_message_update(message_id=message_id, content='Removed mention')
        self.assertEqual(get_apns_badge_count(mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(mentioned_user), 0)

    @mock.patch('zerver.lib.push_notifications.push_notifications_enabled', return_value=True)
    def test_clear_notification_when_group_mention_removed(self, mock_push_notifications: mock.MagicMock) -> None:
        if False:
            return 10
        group_mentioned_user = self.example_user('cordelia')
        self.assertEqual(get_apns_badge_count(group_mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(group_mentioned_user), 0)
        message_id = self._login_and_send_original_stream_message(content='Hello @*hamletcharacters*')
        self.assertEqual(get_apns_badge_count(group_mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(group_mentioned_user), 1)
        self._get_queued_data_for_message_update(message_id=message_id, content='Removed group mention', expect_short_circuit=True)
        self.assertEqual(get_apns_badge_count(group_mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(group_mentioned_user), 0)

    @mock.patch('zerver.lib.push_notifications.push_notifications_enabled', return_value=True)
    def test_not_clear_notification_when_mention_removed_but_stream_notified(self, mock_push_notifications: mock.MagicMock) -> None:
        if False:
            print('Hello World!')
        mentioned_user = self.example_user('iago')
        mentioned_user.enable_stream_push_notifications = True
        mentioned_user.save()
        self.assertEqual(get_apns_badge_count(mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(mentioned_user), 0)
        message_id = self._login_and_send_original_stream_message(content='@**Iago**')
        self.assertEqual(get_apns_badge_count(mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(mentioned_user), 1)
        self._get_queued_data_for_message_update(message_id=message_id, content='Removed mention')
        self.assertEqual(get_apns_badge_count(mentioned_user), 0)
        self.assertEqual(get_apns_badge_count_future(mentioned_user), 1)