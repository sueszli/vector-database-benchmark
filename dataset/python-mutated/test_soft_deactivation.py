from typing import AbstractSet
from unittest import mock
from django.utils.timezone import now as timezone_now
from zerver.actions.alert_words import do_add_alert_words
from zerver.lib.soft_deactivation import add_missing_messages, do_auto_soft_deactivate_users, do_catch_up_soft_deactivated_users, do_soft_activate_users, do_soft_deactivate_user, do_soft_deactivate_users, get_soft_deactivated_users_for_catch_up, get_users_for_soft_deactivation, reactivate_user_if_soft_deactivated
from zerver.lib.stream_subscription import get_subscriptions_for_send_message
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import get_subscription, get_user_messages, make_client
from zerver.models import AlertWord, Client, Message, RealmAuditLog, Stream, UserActivity, UserMessage, UserProfile, get_realm, get_stream
logger_string = 'zulip.soft_deactivation'

class UserSoftDeactivationTests(ZulipTestCase):

    def test_do_soft_deactivate_user(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        self.assertFalse(user.long_term_idle)
        with self.assertLogs(logger_string, level='INFO') as m:
            do_soft_deactivate_user(user)
        self.assertEqual(m.output, [f'INFO:{logger_string}:Soft deactivated user {user.id}'])
        user.refresh_from_db()
        self.assertTrue(user.long_term_idle)

    def test_do_soft_deactivate_users(self) -> None:
        if False:
            return 10
        users = [self.example_user('hamlet'), self.example_user('iago'), self.example_user('cordelia')]
        for user in users:
            self.assertFalse(user.long_term_idle)
        self.send_huddle_message(users[0], users)
        with self.assertLogs(logger_string, level='INFO') as m:
            do_soft_deactivate_users(users)
        log_output = [*(f'INFO:{logger_string}:Soft deactivated user {user.id}' for user in users), f'INFO:{logger_string}:Soft-deactivated batch of {len(users[:100])} users; {len(users[100:])} remain to process']
        self.assertEqual(m.output, log_output)
        for user in users:
            user.refresh_from_db()
            self.assertTrue(user.long_term_idle)

    def test_get_users_for_soft_deactivation(self) -> None:
        if False:
            i = 10
            return i + 15
        users = [self.example_user('hamlet'), self.example_user('iago'), self.example_user('cordelia'), self.example_user('ZOE'), self.example_user('othello'), self.example_user('prospero'), self.example_user('aaron'), self.example_user('polonius'), self.example_user('desdemona'), self.example_user('shiva')]
        (client, _) = Client.objects.get_or_create(name='website')
        query = '/some/random/endpoint'
        last_visit = timezone_now()
        count = 150
        for user_profile in UserProfile.objects.all():
            UserActivity.objects.get_or_create(user_profile=user_profile, client=client, query=query, count=count, last_visit=last_visit)
        filter_kwargs = dict(user_profile__realm=get_realm('zulip'))
        users_to_deactivate = get_users_for_soft_deactivation(-1, filter_kwargs)
        self.assert_length(users_to_deactivate, 10)
        for user in users_to_deactivate:
            self.assertTrue(user in users)

    def test_do_soft_activate_users(self) -> None:
        if False:
            i = 10
            return i + 15
        users = [self.example_user('hamlet'), self.example_user('iago'), self.example_user('cordelia')]
        self.send_huddle_message(users[0], users)
        with self.assertLogs(logger_string, level='INFO') as m:
            do_soft_deactivate_users(users)
        log_output = [*(f'INFO:{logger_string}:Soft deactivated user {user.id}' for user in users), f'INFO:{logger_string}:Soft-deactivated batch of {len(users[:100])} users; {len(users[100:])} remain to process']
        self.assertEqual(m.output, log_output)
        for user in users:
            self.assertTrue(user.long_term_idle)
        with self.assertLogs(logger_string, level='INFO') as m:
            do_soft_activate_users(users)
        log_output = [f'INFO:{logger_string}:Soft reactivated user {user.id}' for user in users]
        self.assertEqual(m.output, log_output)
        for user in users:
            user.refresh_from_db()
            self.assertFalse(user.long_term_idle)

    def test_get_users_for_catch_up(self) -> None:
        if False:
            return 10
        users = [self.example_user('hamlet'), self.example_user('iago'), self.example_user('cordelia'), self.example_user('ZOE'), self.example_user('othello'), self.example_user('prospero'), self.example_user('aaron'), self.example_user('polonius'), self.example_user('desdemona'), self.example_user('shiva')]
        for user_profile in UserProfile.objects.all():
            user_profile.long_term_idle = True
            user_profile.save(update_fields=['long_term_idle'])
        filter_kwargs = dict(realm=get_realm('zulip'))
        users_to_catch_up = get_soft_deactivated_users_for_catch_up(filter_kwargs)
        self.assert_length(users_to_catch_up, 10)
        for user in users_to_catch_up:
            self.assertTrue(user in users)

    def test_do_catch_up_users(self) -> None:
        if False:
            i = 10
            return i + 15
        stream = 'Verona'
        hamlet = self.example_user('hamlet')
        users = [self.example_user('iago'), self.example_user('cordelia')]
        all_users = [*users, hamlet]
        for user in all_users:
            self.subscribe(user, stream)
        with self.assertLogs(logger_string, level='INFO') as m:
            do_soft_deactivate_users(users)
        log_output = [*(f'INFO:{logger_string}:Soft deactivated user {user.id}' for user in users), f'INFO:{logger_string}:Soft-deactivated batch of {len(users[:100])} users; {len(users[100:])} remain to process']
        self.assertEqual(m.output, log_output)
        for user in users:
            self.assertTrue(user.long_term_idle)
        message_id = self.send_stream_message(hamlet, stream, 'Hello world!')
        already_received = UserMessage.objects.filter(message_id=message_id).count()
        with self.assertLogs(logger_string, level='INFO') as m:
            do_catch_up_soft_deactivated_users(users)
        self.assertEqual(m.output, [f'INFO:{logger_string}:Caught up {len(users)} soft-deactivated users'])
        catch_up_received = UserMessage.objects.filter(message_id=message_id).count()
        self.assertEqual(already_received + len(users), catch_up_received)
        for user in users:
            user.refresh_from_db()
            self.assertTrue(user.long_term_idle)
            self.assertEqual(user.last_active_message_id, message_id)

    def test_do_auto_soft_deactivate_users(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        users = [self.example_user('iago'), self.example_user('cordelia'), self.example_user('ZOE'), self.example_user('othello'), self.example_user('prospero'), self.example_user('aaron'), self.example_user('polonius'), self.example_user('desdemona')]
        sender = self.example_user('hamlet')
        realm = get_realm('zulip')
        stream_name = 'announce'
        for user in [*users, sender]:
            self.subscribe(user, stream_name)
        (client, _) = Client.objects.get_or_create(name='website')
        query = '/some/random/endpoint'
        last_visit = timezone_now()
        count = 150
        for user_profile in users:
            UserActivity.objects.get_or_create(user_profile=user_profile, client=client, query=query, count=count, last_visit=last_visit)
        with self.assertLogs(logger_string, level='INFO') as m:
            users_deactivated = do_auto_soft_deactivate_users(-1, realm)
        log_output = [*(f'INFO:{logger_string}:Soft deactivated user {user.id}' for user in users), f'INFO:{logger_string}:Soft-deactivated batch of {len(users[:100])} users; {len(users[100:])} remain to process', f'INFO:{logger_string}:Caught up {len(users)} soft-deactivated users']
        self.assertEqual(set(m.output), set(log_output))
        self.assert_length(users_deactivated, len(users))
        for user in users:
            self.assertTrue(user in users_deactivated)
        message_id = self.send_stream_message(sender, stream_name)
        received_count = UserMessage.objects.filter(user_profile__in=users, message_id=message_id).count()
        self.assertEqual(0, received_count)
        with self.assertLogs(logger_string, level='INFO') as m:
            users_deactivated = do_auto_soft_deactivate_users(-1, realm)
        log_output = [f'INFO:{logger_string}:Caught up {len(users)} soft-deactivated users']
        self.assertEqual(set(m.output), set(log_output))
        self.assert_length(users_deactivated, 0)
        received_count = UserMessage.objects.filter(user_profile__in=users, message_id=message_id).count()
        self.assert_length(users, received_count)
        message_id = self.send_stream_message(sender, stream_name)
        received_count = UserMessage.objects.filter(user_profile__in=users, message_id=message_id).count()
        self.assertEqual(0, received_count)
        with self.settings(AUTO_CATCH_UP_SOFT_DEACTIVATED_USERS=False):
            with self.assertLogs(logger_string, level='INFO') as m:
                users_deactivated = do_auto_soft_deactivate_users(-1, realm)
        self.assertEqual(m.output, [f'INFO:{logger_string}:Not catching up users since AUTO_CATCH_UP_SOFT_DEACTIVATED_USERS is off'])
        self.assert_length(users_deactivated, 0)
        received_count = UserMessage.objects.filter(user_profile__in=users, message_id=message_id).count()
        self.assertEqual(0, received_count)

class SoftDeactivationMessageTest(ZulipTestCase):

    def test_reactivate_user_if_soft_deactivated(self) -> None:
        if False:
            i = 10
            return i + 15
        recipient_list = [self.example_user('hamlet'), self.example_user('iago')]
        for user_profile in recipient_list:
            self.subscribe(user_profile, 'Denmark')
        sender = self.example_user('iago')
        stream_name = 'Denmark'
        topic_name = 'foo'

        def last_realm_audit_log_entry(event_type: int) -> RealmAuditLog:
            if False:
                print('Hello World!')
            return RealmAuditLog.objects.filter(event_type=event_type).order_by('-event_time')[0]
        long_term_idle_user = self.example_user('hamlet')
        self.send_stream_message(long_term_idle_user, stream_name)
        with self.assertLogs(logger_string, level='INFO') as info_logs:
            do_soft_deactivate_users([long_term_idle_user])
        self.assertEqual(info_logs.output, [f'INFO:{logger_string}:Soft deactivated user {long_term_idle_user.id}', f'INFO:{logger_string}:Soft-deactivated batch of 1 users; 0 remain to process'])
        message = 'Test message 1'
        message_id = self.send_stream_message(sender, stream_name, message, topic_name)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        self.assertNotEqual(idle_user_msg_list[-1].content, message)
        with self.assert_database_query_count(7):
            reactivate_user_if_soft_deactivated(long_term_idle_user)
        self.assertFalse(long_term_idle_user.long_term_idle)
        self.assertEqual(last_realm_audit_log_entry(RealmAuditLog.USER_SOFT_ACTIVATED).modified_user, long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count + 1)
        self.assertEqual(idle_user_msg_list[-1].content, message)
        long_term_idle_user.refresh_from_db()
        self.assertEqual(long_term_idle_user.last_active_message_id, message_id)

    def test_add_missing_messages(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        recipient_list = [self.example_user('hamlet'), self.example_user('iago')]
        for user_profile in recipient_list:
            self.subscribe(user_profile, 'Denmark')
        sender = self.example_user('iago')
        realm = sender.realm
        sending_client = make_client(name='test suite')
        stream_name = 'Denmark'
        stream = get_stream(stream_name, realm)
        topic_name = 'foo'

        def send_fake_message(message_content: str, stream: Stream) -> Message:
            if False:
                return 10
            '\n            The purpose of this helper is to create a Message object without corresponding\n            UserMessage rows being created.\n            '
            recipient = stream.recipient
            message = Message(sender=sender, realm=realm, recipient=recipient, content=message_content, date_sent=timezone_now(), sending_client=sending_client)
            message.set_topic_name(topic_name)
            message.save()
            return message
        long_term_idle_user = self.example_user('hamlet')
        self.send_stream_message(long_term_idle_user, stream_name)
        with self.assertLogs(logger_string, level='INFO') as info_logs:
            do_soft_deactivate_users([long_term_idle_user])
        self.assertEqual(info_logs.output, [f'INFO:{logger_string}:Soft deactivated user {long_term_idle_user.id}', f'INFO:{logger_string}:Soft-deactivated batch of 1 users; 0 remain to process'])
        sent_message = send_fake_message('Test message 1', stream)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        self.assertNotEqual(idle_user_msg_list[-1], sent_message)
        with self.assert_database_query_count(5):
            add_missing_messages(long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count + 1)
        self.assertEqual(idle_user_msg_list[-1], sent_message)
        long_term_idle_user.refresh_from_db()
        self.assertEqual(long_term_idle_user.last_active_message_id, sent_message.id)
        sent_message = send_fake_message('Test message 2', stream)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        self.assertNotEqual(idle_user_msg_list[-1], sent_message)
        with self.assert_database_query_count(6):
            add_missing_messages(long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count + 1)
        self.assertEqual(idle_user_msg_list[-1], sent_message)
        long_term_idle_user.refresh_from_db()
        self.assertEqual(long_term_idle_user.last_active_message_id, sent_message.id)
        sent_message_list = []
        sent_message_list.append(send_fake_message('Test message 3', stream))
        self.unsubscribe(long_term_idle_user, stream_name)
        send_fake_message('Test message 4', stream)
        self.subscribe(long_term_idle_user, stream_name)
        sent_message_list.append(send_fake_message('Test message 5', stream))
        sent_message_list.reverse()
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        for sent_message in sent_message_list:
            self.assertNotEqual(idle_user_msg_list.pop(), sent_message)
        with self.assert_database_query_count(5):
            add_missing_messages(long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count + 2)
        for sent_message in sent_message_list:
            self.assertEqual(idle_user_msg_list.pop(), sent_message)
        long_term_idle_user.refresh_from_db()
        self.assertEqual(long_term_idle_user.last_active_message_id, sent_message_list[0].id)
        sent_message_list = []
        sent_message_list.append(send_fake_message('Test message 6', stream))
        self.unsubscribe(long_term_idle_user, stream_name)
        self.subscribe(long_term_idle_user, stream_name)
        sent_message_list.append(send_fake_message('Test message 7', stream))
        self.unsubscribe(long_term_idle_user, stream_name)
        send_fake_message('Test message 8', stream)
        self.subscribe(long_term_idle_user, stream_name)
        self.unsubscribe(long_term_idle_user, stream_name)
        sent_message_list.reverse()
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        for sent_message in sent_message_list:
            self.assertNotEqual(idle_user_msg_list.pop(), sent_message)
        with self.assert_database_query_count(5):
            add_missing_messages(long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count + 2)
        for sent_message in sent_message_list:
            self.assertEqual(idle_user_msg_list.pop(), sent_message)
        long_term_idle_user.refresh_from_db()
        self.assertEqual(long_term_idle_user.last_active_message_id, sent_message_list[0].id)
        do_soft_activate_users([long_term_idle_user])
        self.subscribe(long_term_idle_user, stream_name)
        sent_message_id = self.send_stream_message(sender, stream_name, 'Test message 9')
        self.unsubscribe(long_term_idle_user, stream_name)
        with self.assertLogs(logger_string, level='INFO') as info_logs:
            do_soft_deactivate_users([long_term_idle_user])
        self.assertEqual(info_logs.output, [f'INFO:{logger_string}:Soft deactivated user {long_term_idle_user.id}', f'INFO:{logger_string}:Soft-deactivated batch of 1 users; 0 remain to process'])
        send_fake_message('Test message 10', stream)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        self.assertEqual(idle_user_msg_list[-1].id, sent_message_id)
        with self.assert_database_query_count(3):
            add_missing_messages(long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count)
        stream_name = 'Core'
        private_stream = self.make_stream('Core', invite_only=True)
        self.subscribe(self.example_user('iago'), stream_name)
        sent_message_list = []
        send_fake_message('Test message 11', private_stream)
        self.subscribe(self.example_user('hamlet'), stream_name)
        sent_message_list.append(send_fake_message('Test message 12', private_stream))
        self.unsubscribe(long_term_idle_user, stream_name)
        send_fake_message('Test message 13', private_stream)
        self.subscribe(long_term_idle_user, stream_name)
        sent_message_list.append(send_fake_message('Test message 14', private_stream))
        sent_message_list.reverse()
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        for sent_message in sent_message_list:
            self.assertNotEqual(idle_user_msg_list.pop(), sent_message)
        with self.assert_database_query_count(5):
            add_missing_messages(long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count + 2)
        for sent_message in sent_message_list:
            self.assertEqual(idle_user_msg_list.pop(), sent_message)
        long_term_idle_user.refresh_from_db()
        self.assertEqual(long_term_idle_user.last_active_message_id, sent_message_list[0].id)

    @mock.patch('zerver.lib.soft_deactivation.BULK_CREATE_BATCH_SIZE', 2)
    def test_add_missing_messages_pagination(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        recipient_list = [self.example_user('hamlet'), self.example_user('iago')]
        stream_name = 'Denmark'
        for user_profile in recipient_list:
            self.subscribe(user_profile, stream_name)
        sender = self.example_user('iago')
        long_term_idle_user = self.example_user('hamlet')
        self.send_stream_message(long_term_idle_user, stream_name)
        with self.assertLogs(logger_string, level='INFO') as info_logs:
            do_soft_deactivate_users([long_term_idle_user])
        self.assertEqual(info_logs.output, [f'INFO:{logger_string}:Soft deactivated user {long_term_idle_user.id}', f'INFO:{logger_string}:Soft-deactivated batch of 1 users; 0 remain to process'])
        num_new_messages = 5
        message_ids = []
        for _ in range(num_new_messages):
            message_id = self.send_stream_message(sender, stream_name)
            message_ids.append(message_id)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        idle_user_msg_count = len(idle_user_msg_list)
        with self.assert_database_query_count(9):
            add_missing_messages(long_term_idle_user)
        idle_user_msg_list = get_user_messages(long_term_idle_user)
        self.assert_length(idle_user_msg_list, idle_user_msg_count + num_new_messages)
        long_term_idle_user.refresh_from_db()
        self.assertEqual(long_term_idle_user.last_active_message_id, message_ids[-1])

    def test_user_message_filter(self) -> None:
        if False:
            return 10
        AlertWord.objects.all().delete()
        long_term_idle_user = self.example_user('hamlet')
        cordelia = self.example_user('cordelia')
        sender = self.example_user('iago')
        stream_name = 'Brand New Stream'
        topic_name = 'foo'
        realm_id = cordelia.realm_id
        self.subscribe(long_term_idle_user, stream_name)
        self.subscribe(cordelia, stream_name)
        self.subscribe(sender, stream_name)
        stream_id = get_stream(stream_name, cordelia.realm).id

        def send_stream_message(content: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.send_stream_message(sender, stream_name, content, topic_name)

        def send_personal_message(content: str) -> None:
            if False:
                i = 10
                return i + 15
            self.send_personal_message(sender, self.example_user('hamlet'), content)
        self.send_stream_message(long_term_idle_user, stream_name)
        with self.assertLogs(logger_string, level='INFO') as info_logs:
            do_soft_deactivate_users([long_term_idle_user])
        self.assertEqual(info_logs.output, [f'INFO:{logger_string}:Soft deactivated user {long_term_idle_user.id}', f'INFO:{logger_string}:Soft-deactivated batch of 1 users; 0 remain to process'])

        def assert_um_count(user: UserProfile, count: int) -> None:
            if False:
                return 10
            user_messages = get_user_messages(user)
            self.assert_length(user_messages, count)

        def assert_last_um_content(user: UserProfile, content: str, negate: bool=False) -> None:
            if False:
                i = 10
                return i + 15
            user_messages = get_user_messages(user)
            if negate:
                self.assertNotEqual(user_messages[-1].content, content)
            else:
                self.assertEqual(user_messages[-1].content, content)

        def assert_num_possible_users(expected_count: int, *, possible_stream_wildcard_mention: bool=False, topic_participant_user_ids: AbstractSet[int]=set(), possibly_mentioned_user_ids: AbstractSet[int]=set()) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(len(get_subscriptions_for_send_message(realm_id=realm_id, stream_id=stream_id, topic_name=topic_name, possible_stream_wildcard_mention=possible_stream_wildcard_mention, topic_participant_user_ids=topic_participant_user_ids, possibly_mentioned_user_ids=possibly_mentioned_user_ids)), expected_count)

        def assert_stream_message_sent_to_idle_user(content: str, *, possible_stream_wildcard_mention: bool=False, topic_participant_user_ids: AbstractSet[int]=set(), possibly_mentioned_user_ids: AbstractSet[int]=set()) -> None:
            if False:
                for i in range(10):
                    print('nop')
            assert_num_possible_users(expected_count=3, possible_stream_wildcard_mention=possible_stream_wildcard_mention, topic_participant_user_ids=topic_participant_user_ids, possibly_mentioned_user_ids=possibly_mentioned_user_ids)
            general_user_msg_count = len(get_user_messages(cordelia))
            soft_deactivated_user_msg_count = len(get_user_messages(long_term_idle_user))
            send_stream_message(content)
            assert_um_count(long_term_idle_user, soft_deactivated_user_msg_count + 1)
            assert_um_count(cordelia, general_user_msg_count + 1)
            assert_last_um_content(long_term_idle_user, content)
            assert_last_um_content(cordelia, content)

        def assert_stream_message_not_sent_to_idle_user(content: str, *, possibly_mentioned_user_ids: AbstractSet[int]=set(), false_alarm_row: bool=False) -> None:
            if False:
                i = 10
                return i + 15
            if false_alarm_row:
                assert_num_possible_users(3, possibly_mentioned_user_ids=possibly_mentioned_user_ids)
            else:
                assert_num_possible_users(2)
            general_user_msg_count = len(get_user_messages(cordelia))
            soft_deactivated_user_msg_count = len(get_user_messages(long_term_idle_user))
            send_stream_message(content)
            assert_um_count(long_term_idle_user, soft_deactivated_user_msg_count)
            assert_um_count(cordelia, general_user_msg_count + 1)
            assert_last_um_content(long_term_idle_user, content, negate=True)
            assert_last_um_content(cordelia, content)
        assert_stream_message_not_sent_to_idle_user('Test message 1')
        sub = get_subscription(stream_name, long_term_idle_user)
        sub.push_notifications = True
        sub.save()
        assert_stream_message_sent_to_idle_user('Sub push')
        sub.push_notifications = False
        sub.save()
        assert_stream_message_not_sent_to_idle_user('Sub no push')
        sub.push_notifications = None
        sub.save()
        long_term_idle_user.enable_stream_push_notifications = True
        long_term_idle_user.save()
        assert_stream_message_sent_to_idle_user('User push')
        long_term_idle_user.enable_stream_push_notifications = False
        long_term_idle_user.save()
        assert_stream_message_not_sent_to_idle_user('User no push')
        sub.email_notifications = True
        sub.save()
        assert_stream_message_sent_to_idle_user('Sub email')
        sub.email_notifications = False
        sub.save()
        assert_stream_message_not_sent_to_idle_user('Sub no email')
        sub.email_notifications = None
        sub.save()
        long_term_idle_user.enable_stream_email_notifications = True
        long_term_idle_user.save()
        assert_stream_message_sent_to_idle_user('User email')
        long_term_idle_user.enable_stream_email_notifications = False
        long_term_idle_user.save()
        assert_stream_message_not_sent_to_idle_user('User no email')
        soft_deactivated_user_msg_count = len(get_user_messages(long_term_idle_user))
        message = 'Test direct message'
        send_personal_message(message)
        assert_um_count(long_term_idle_user, soft_deactivated_user_msg_count + 1)
        assert_last_um_content(long_term_idle_user, message)
        assert_stream_message_sent_to_idle_user('Test @**King Hamlet** mention', possibly_mentioned_user_ids={long_term_idle_user.id})
        assert_stream_message_not_sent_to_idle_user('Test `@**King Hamlet**` mention', possibly_mentioned_user_ids={long_term_idle_user.id}, false_alarm_row=True)
        assert_stream_message_not_sent_to_idle_user("Test @**Cordelia, Lear's daughter**  mention")
        assert_stream_message_sent_to_idle_user('Test @**all** mention', possible_stream_wildcard_mention=True)
        assert_stream_message_sent_to_idle_user('Test @**everyone** mention', possible_stream_wildcard_mention=True)
        assert_stream_message_sent_to_idle_user('Test @**stream** mention', possible_stream_wildcard_mention=True)
        assert_stream_message_not_sent_to_idle_user('Test @**bogus** mention')
        do_soft_activate_users([long_term_idle_user])
        self.send_stream_message(long_term_idle_user, stream_name, 'Hi', topic_name)
        topic_participant_user_ids = {long_term_idle_user.id}
        do_soft_deactivate_users([long_term_idle_user])
        assert_stream_message_sent_to_idle_user('Test @**topic** mention', topic_participant_user_ids=topic_participant_user_ids)
        do_add_alert_words(long_term_idle_user, ['test_alert_word'])
        assert_stream_message_sent_to_idle_user('Testing test_alert_word')
        do_add_alert_words(cordelia, ['cordelia'])
        assert_stream_message_not_sent_to_idle_user('cordelia', false_alarm_row=True)
        assert_stream_message_not_sent_to_idle_user('/me says test', false_alarm_row=True)
        AlertWord.objects.filter(user_profile=long_term_idle_user).delete()
        assert_stream_message_not_sent_to_idle_user('no alert words')