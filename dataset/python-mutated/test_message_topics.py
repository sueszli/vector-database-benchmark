from unittest import mock
from django.utils.timezone import now as timezone_now
from zerver.actions.streams import do_change_stream_permission
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import timeout_mock
from zerver.lib.timeout import TimeoutExpiredError
from zerver.models import Message, UserMessage, get_client, get_realm, get_stream

class TopicHistoryTest(ZulipTestCase):

    def test_topics_history_zephyr_mirror(self) -> None:
        if False:
            print('Hello World!')
        user_profile = self.mit_user('sipbtest')
        stream_name = 'new_stream'
        self.subscribe(self.mit_user('starnine'), stream_name)
        stream = get_stream(stream_name, user_profile.realm)
        self.send_stream_message(self.mit_user('starnine'), stream_name, topic_name='secret topic')
        self.login_user(user_profile)
        self.subscribe(user_profile, stream_name)
        endpoint = f'/json/users/me/{stream.id}/topics'
        result = self.client_get(endpoint, {}, subdomain='zephyr')
        history = self.assert_json_success(result)['topics']
        self.assertEqual(history, [])

    def test_topics_history(self) -> None:
        if False:
            print('Hello World!')
        user_profile = self.example_user('iago')
        self.login_user(user_profile)
        stream_name = 'Verona'
        stream = get_stream(stream_name, user_profile.realm)
        recipient = stream.recipient

        def create_test_message(topic: str) -> int:
            if False:
                while True:
                    i = 10
            hamlet = self.example_user('hamlet')
            message = Message(sender=hamlet, recipient=recipient, realm=stream.realm, content='whatever', date_sent=timezone_now(), sending_client=get_client('whatever'))
            message.set_topic_name(topic)
            message.save()
            UserMessage.objects.create(user_profile=user_profile, message=message, flags=0)
            return message.id
        create_test_message('topic2')
        create_test_message('toPIc1')
        create_test_message('toPIc0')
        create_test_message('topic2')
        create_test_message('topic2')
        create_test_message('Topic2')
        topic2_msg_id = create_test_message('topic2')
        create_test_message('topic1')
        create_test_message('topic1')
        topic1_msg_id = create_test_message('topic1')
        topic0_msg_id = create_test_message('topic0')
        endpoint = f'/json/users/me/{stream.id}/topics'
        result = self.client_get(endpoint, {})
        history = self.assert_json_success(result)['topics']
        history = history[:3]
        self.assertEqual([topic['name'] for topic in history], ['topic0', 'topic1', 'topic2'])
        self.assertEqual([topic['max_id'] for topic in history], [topic0_msg_id, topic1_msg_id, topic2_msg_id])
        self.login('cordelia')
        result = self.client_get(endpoint, {})
        history = self.assert_json_success(result)['topics']
        history = history[:3]
        self.assertEqual([topic['name'] for topic in history], ['topic0', 'topic1', 'topic2'])
        self.assertIn('topic0', [topic['name'] for topic in history])
        self.assertEqual([topic['max_id'] for topic in history], [topic0_msg_id, topic1_msg_id, topic2_msg_id])
        do_change_stream_permission(stream, invite_only=True, history_public_to_subscribers=False, is_web_public=False, acting_user=self.example_user('cordelia'))
        self.subscribe(self.example_user('cordelia'), stream.name)
        result = self.client_get(endpoint, {})
        history = self.assert_json_success(result)['topics']
        history = history[:3]
        self.assertNotIn('topic0', [topic['name'] for topic in history])
        self.assertNotIn('topic1', [topic['name'] for topic in history])
        self.assertNotIn('topic2', [topic['name'] for topic in history])

    def test_bad_stream_id(self) -> None:
        if False:
            print('Hello World!')
        self.login('iago')
        endpoint = '/json/users/me/9999999999/topics'
        result = self.client_get(endpoint, {})
        self.assert_json_error(result, 'Invalid stream ID')
        bad_stream = self.make_stream('mit_stream', realm=get_realm('zephyr'))
        endpoint = f'/json/users/me/{bad_stream.id}/topics'
        result = self.client_get(endpoint, {})
        self.assert_json_error(result, 'Invalid stream ID')
        private_stream = self.make_stream('private_stream', invite_only=True)
        endpoint = f'/json/users/me/{private_stream.id}/topics'
        result = self.client_get(endpoint, {})
        self.assert_json_error(result, 'Invalid stream ID')

    def test_get_topics_web_public_stream_web_public_request(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        iago = self.example_user('iago')
        stream = self.make_stream('web-public-stream', is_web_public=True)
        self.subscribe(iago, stream.name)
        for i in range(3):
            self.send_stream_message(iago, stream.name, topic_name='topic' + str(i))
        endpoint = f'/json/users/me/{stream.id}/topics'
        result = self.client_get(endpoint)
        history = self.assert_json_success(result)['topics']
        self.assertEqual([topic['name'] for topic in history], ['topic2', 'topic1', 'topic0'])

    def test_get_topics_non_web_public_stream_web_public_request(self) -> None:
        if False:
            i = 10
            return i + 15
        stream = get_stream('Verona', self.example_user('iago').realm)
        endpoint = f'/json/users/me/{stream.id}/topics'
        result = self.client_get(endpoint)
        self.assert_json_error(result, 'Invalid stream ID', 400)

    def test_get_topics_non_existent_stream_web_public_request(self) -> None:
        if False:
            while True:
                i = 10
        non_existent_stream_id = 10000000000000000000000
        endpoint = f'/json/users/me/{non_existent_stream_id}/topics'
        result = self.client_get(endpoint)
        self.assert_json_error(result, 'Invalid stream ID', 400)

class TopicDeleteTest(ZulipTestCase):

    def test_topic_delete(self) -> None:
        if False:
            i = 10
            return i + 15
        initial_last_msg_id = self.get_last_message().id
        stream_name = 'new_stream'
        topic_name = 'new topic 2'
        user_profile = self.example_user('hamlet')
        self.subscribe(user_profile, stream_name)
        stream = get_stream(stream_name, user_profile.realm)
        self.send_stream_message(user_profile, stream_name, topic_name=topic_name)
        last_msg_id = self.send_stream_message(user_profile, stream_name, topic_name=topic_name)
        self.login_user(user_profile)
        endpoint = '/json/streams/' + str(stream.id) + '/delete_topic'
        result = self.client_post(endpoint, {'topic_name': topic_name})
        self.assert_json_error(result, 'Must be an organization administrator')
        self.assertTrue(Message.objects.filter(id=last_msg_id).exists())
        do_change_stream_permission(stream, invite_only=True, history_public_to_subscribers=False, is_web_public=False, acting_user=user_profile)
        user_profile = self.example_user('iago')
        self.subscribe(user_profile, stream_name)
        self.login_user(user_profile)
        new_last_msg_id = self.send_stream_message(user_profile, stream_name, topic_name=topic_name)
        self.assertEqual(self.get_last_message().id, new_last_msg_id)
        result = self.client_post(endpoint, {'topic_name': topic_name})
        result_dict = self.assert_json_success(result)
        self.assertTrue(result_dict['complete'])
        self.assertTrue(Message.objects.filter(id=last_msg_id).exists())
        result = self.client_post(endpoint, {'topic_name': topic_name})
        result_dict = self.assert_json_success(result)
        self.assertTrue(result_dict['complete'])
        self.assertTrue(Message.objects.filter(id=last_msg_id).exists())
        do_change_stream_permission(stream, invite_only=True, history_public_to_subscribers=True, is_web_public=False, acting_user=user_profile)
        with timeout_mock('zerver.views.streams'):
            result = self.client_post(endpoint, {'topic_name': topic_name})
        result_dict = self.assert_json_success(result)
        self.assertTrue(result_dict['complete'])
        self.assertFalse(Message.objects.filter(id=last_msg_id).exists())
        self.assertTrue(Message.objects.filter(id=initial_last_msg_id).exists())
        with timeout_mock('zerver.views.streams'):
            result = self.client_post(endpoint, {'topic_name': topic_name})
        result_dict = self.assert_json_success(result)
        self.assertTrue(result_dict['complete'])
        self.assertFalse(Message.objects.filter(id=last_msg_id).exists())
        self.assertTrue(Message.objects.filter(id=initial_last_msg_id).exists())

    def test_topic_delete_timeout(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        stream_name = 'new_stream'
        topic_name = 'new topic 2'
        user_profile = self.example_user('iago')
        self.subscribe(user_profile, stream_name)
        stream = get_stream(stream_name, user_profile.realm)
        self.send_stream_message(user_profile, stream_name, topic_name=topic_name)
        self.login_user(user_profile)
        endpoint = '/json/streams/' + str(stream.id) + '/delete_topic'
        with mock.patch('zerver.views.streams.timeout', side_effect=TimeoutExpiredError):
            result = self.client_post(endpoint, {'topic_name': topic_name})
            result_dict = self.assert_json_success(result)
            self.assertFalse(result_dict['complete'])