import datetime
import re
import time
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Union
from unittest import mock
import orjson
import time_machine
from django.utils.timezone import now as timezone_now
from zerver.actions.scheduled_messages import SCHEDULED_MESSAGE_LATE_CUTOFF_MINUTES, try_deliver_one_scheduled_message
from zerver.actions.users import change_user_is_active
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import most_recent_message
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.models import Attachment, Message, Recipient, ScheduledMessage, UserMessage
if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse

class ScheduledMessageTest(ZulipTestCase):

    def last_scheduled_message(self) -> ScheduledMessage:
        if False:
            for i in range(10):
                print('nop')
        return ScheduledMessage.objects.all().order_by('-id')[0]

    def get_scheduled_message(self, id: str) -> ScheduledMessage:
        if False:
            while True:
                i = 10
        return ScheduledMessage.objects.get(id=id)

    def do_schedule_message(self, msg_type: str, to: Union[int, List[str], List[int]], msg: str, scheduled_delivery_timestamp: int) -> 'TestHttpResponse':
        if False:
            return 10
        self.login('hamlet')
        topic_name = ''
        if msg_type == 'stream':
            topic_name = 'Test topic'
        payload = {'type': msg_type, 'to': orjson.dumps(to).decode(), 'content': msg, 'topic': topic_name, 'scheduled_delivery_timestamp': scheduled_delivery_timestamp}
        result = self.client_post('/json/scheduled_messages', payload)
        return result

    def test_schedule_message(self) -> None:
        if False:
            return 10
        content = 'Test message'
        scheduled_delivery_timestamp = int(time.time() + 86400)
        verona_stream_id = self.get_stream_id('Verona')
        result = self.do_schedule_message('stream', verona_stream_id, content + ' 1', scheduled_delivery_timestamp)
        scheduled_message = self.last_scheduled_message()
        self.assert_json_success(result)
        self.assertEqual(scheduled_message.content, 'Test message 1')
        self.assertEqual(scheduled_message.rendered_content, '<p>Test message 1</p>')
        self.assertEqual(scheduled_message.topic_name(), 'Test topic')
        self.assertEqual(scheduled_message.scheduled_timestamp, timestamp_to_datetime(scheduled_delivery_timestamp))
        othello = self.example_user('othello')
        result = self.do_schedule_message('direct', [othello.id], content + ' 3', scheduled_delivery_timestamp)
        scheduled_message = self.last_scheduled_message()
        self.assert_json_success(result)
        self.assertEqual(scheduled_message.content, 'Test message 3')
        self.assertEqual(scheduled_message.rendered_content, '<p>Test message 3</p>')
        self.assertEqual(scheduled_message.scheduled_timestamp, timestamp_to_datetime(scheduled_delivery_timestamp))
        result = self.do_schedule_message('direct', [othello.email], content + ' 4', scheduled_delivery_timestamp)
        self.assert_json_error(result, 'Recipient list may only contain user IDs')

    def create_scheduled_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        content = 'Test message'
        scheduled_delivery_datetime = timezone_now() + datetime.timedelta(minutes=5)
        scheduled_delivery_timestamp = int(scheduled_delivery_datetime.timestamp())
        verona_stream_id = self.get_stream_id('Verona')
        result = self.do_schedule_message('stream', verona_stream_id, content + ' 1', scheduled_delivery_timestamp)
        self.assert_json_success(result)

    def test_successful_deliver_stream_scheduled_message(self) -> None:
        if False:
            i = 10
            return i + 15
        logger = mock.Mock()
        result = try_deliver_one_scheduled_message(logger)
        self.assertFalse(result)
        self.create_scheduled_message()
        scheduled_message = self.last_scheduled_message()
        more_than_scheduled_delivery_datetime = scheduled_message.scheduled_timestamp + datetime.timedelta(minutes=1)
        with time_machine.travel(more_than_scheduled_delivery_datetime, tick=False):
            result = try_deliver_one_scheduled_message(logger)
            self.assertTrue(result)
            logger.info.assert_called_once_with('Sending scheduled message %s with date %s (sender: %s)', scheduled_message.id, scheduled_message.scheduled_timestamp, scheduled_message.sender_id)
            scheduled_message.refresh_from_db()
            assert isinstance(scheduled_message.delivered_message_id, int)
            self.assertEqual(scheduled_message.delivered, True)
            self.assertEqual(scheduled_message.failed, False)
            delivered_message = Message.objects.get(id=scheduled_message.delivered_message_id)
            self.assertEqual(delivered_message.content, scheduled_message.content)
            self.assertEqual(delivered_message.rendered_content, scheduled_message.rendered_content)
            self.assertEqual(delivered_message.topic_name(), scheduled_message.topic_name())
            self.assertEqual(delivered_message.date_sent, more_than_scheduled_delivery_datetime)

    def test_successful_deliver_direct_scheduled_message(self) -> None:
        if False:
            print('Hello World!')
        logger = mock.Mock()
        self.assertFalse(try_deliver_one_scheduled_message(logger))
        content = 'Test message'
        scheduled_delivery_datetime = timezone_now() + datetime.timedelta(minutes=5)
        scheduled_delivery_timestamp = int(scheduled_delivery_datetime.timestamp())
        sender = self.example_user('hamlet')
        othello = self.example_user('othello')
        response = self.do_schedule_message('direct', [othello.id], content + ' 3', scheduled_delivery_timestamp)
        self.assert_json_success(response)
        scheduled_message = self.last_scheduled_message()
        more_than_scheduled_delivery_datetime = scheduled_delivery_datetime + datetime.timedelta(minutes=1)
        with time_machine.travel(more_than_scheduled_delivery_datetime, tick=False):
            result = try_deliver_one_scheduled_message(logger)
            self.assertTrue(result)
            logger.info.assert_called_once_with('Sending scheduled message %s with date %s (sender: %s)', scheduled_message.id, scheduled_message.scheduled_timestamp, scheduled_message.sender_id)
            scheduled_message.refresh_from_db()
            assert isinstance(scheduled_message.delivered_message_id, int)
            self.assertEqual(scheduled_message.delivered, True)
            self.assertEqual(scheduled_message.failed, False)
            delivered_message = Message.objects.get(id=scheduled_message.delivered_message_id)
            self.assertEqual(delivered_message.content, scheduled_message.content)
            self.assertEqual(delivered_message.rendered_content, scheduled_message.rendered_content)
            self.assertEqual(delivered_message.date_sent, more_than_scheduled_delivery_datetime)
            sender_user_message = UserMessage.objects.get(message_id=scheduled_message.delivered_message_id, user_profile_id=sender.id)
            self.assertTrue(sender_user_message.flags.read)
        new_delivery_datetime = timezone_now() + datetime.timedelta(minutes=7)
        new_delivery_timestamp = int(new_delivery_datetime.timestamp())
        content = 'New message content'
        payload = {'content': content, 'scheduled_delivery_timestamp': new_delivery_timestamp}
        updated_response = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload)
        self.assert_json_error(updated_response, 'Scheduled message was already sent')

    def test_successful_deliver_direct_scheduled_message_to_self(self) -> None:
        if False:
            print('Hello World!')
        logger = mock.Mock()
        self.assertFalse(try_deliver_one_scheduled_message(logger))
        content = 'Test message to self'
        scheduled_delivery_datetime = timezone_now() + datetime.timedelta(minutes=5)
        scheduled_delivery_timestamp = int(scheduled_delivery_datetime.timestamp())
        sender = self.example_user('hamlet')
        response = self.do_schedule_message('direct', [sender.id], content, scheduled_delivery_timestamp)
        self.assert_json_success(response)
        scheduled_message = self.last_scheduled_message()
        more_than_scheduled_delivery_datetime = scheduled_delivery_datetime + datetime.timedelta(minutes=1)
        with time_machine.travel(more_than_scheduled_delivery_datetime, tick=False):
            result = try_deliver_one_scheduled_message(logger)
            self.assertTrue(result)
            logger.info.assert_called_once_with('Sending scheduled message %s with date %s (sender: %s)', scheduled_message.id, scheduled_message.scheduled_timestamp, scheduled_message.sender_id)
            scheduled_message.refresh_from_db()
            assert isinstance(scheduled_message.delivered_message_id, int)
            self.assertEqual(scheduled_message.delivered, True)
            self.assertEqual(scheduled_message.failed, False)
            delivered_message = Message.objects.get(id=scheduled_message.delivered_message_id)
            self.assertEqual(delivered_message.content, scheduled_message.content)
            self.assertEqual(delivered_message.rendered_content, scheduled_message.rendered_content)
            self.assertEqual(delivered_message.date_sent, more_than_scheduled_delivery_datetime)
            sender_user_message = UserMessage.objects.get(message_id=scheduled_message.delivered_message_id, user_profile_id=sender.id)
            self.assertFalse(sender_user_message.flags.read)

    def verify_deliver_scheduled_message_failure(self, scheduled_message: ScheduledMessage, logger: mock.Mock, expected_failure_message: str) -> None:
        if False:
            return 10
        result = try_deliver_one_scheduled_message(logger)
        self.assertTrue(result)
        scheduled_message.refresh_from_db()
        self.assertEqual(scheduled_message.failure_message, expected_failure_message)
        calls = [mock.call('Sending scheduled message %s with date %s (sender: %s)', scheduled_message.id, scheduled_message.scheduled_timestamp, scheduled_message.sender_id), mock.call('Failed with message: %s', scheduled_message.failure_message)]
        logger.info.assert_has_calls(calls)
        self.assertEqual(logger.info.call_count, 2)
        self.assertTrue(scheduled_message.failed)

    def test_too_late_to_deliver_scheduled_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expected_failure_message = 'Message could not be sent at the scheduled time.'
        logger = mock.Mock()
        self.create_scheduled_message()
        scheduled_message = self.last_scheduled_message()
        too_late_to_send_message_datetime = scheduled_message.scheduled_timestamp + datetime.timedelta(minutes=SCHEDULED_MESSAGE_LATE_CUTOFF_MINUTES + 1)
        with time_machine.travel(too_late_to_send_message_datetime, tick=False):
            self.verify_deliver_scheduled_message_failure(scheduled_message, logger, expected_failure_message)
        realm = scheduled_message.realm
        msg = most_recent_message(scheduled_message.sender)
        self.assertEqual(msg.recipient.type, msg.recipient.PERSONAL)
        self.assertEqual(msg.sender_id, self.notification_bot(realm).id)
        self.assertIn(expected_failure_message, msg.content)

    def test_realm_deactivated_failed_to_deliver_scheduled_message(self) -> None:
        if False:
            return 10
        expected_failure_message = 'This organization has been deactivated'
        logger = mock.Mock()
        self.create_scheduled_message()
        scheduled_message = self.last_scheduled_message()
        self.assertFalse(scheduled_message.realm.deactivated)
        message_before_deactivation = most_recent_message(scheduled_message.sender)
        more_than_scheduled_delivery_datetime = scheduled_message.scheduled_timestamp + datetime.timedelta(minutes=1)
        with time_machine.travel(more_than_scheduled_delivery_datetime, tick=False):
            scheduled_message = self.last_scheduled_message()
            scheduled_message.realm.deactivated = True
            scheduled_message.realm.save()
            self.verify_deliver_scheduled_message_failure(scheduled_message, logger, expected_failure_message)
        self.assertTrue(scheduled_message.realm.deactivated)
        message_after_deactivation = most_recent_message(scheduled_message.sender)
        self.assertEqual(message_after_deactivation.content, message_before_deactivation.content)
        self.assertNotIn(expected_failure_message, message_after_deactivation.content)

    def test_sender_deactivated_failed_to_deliver_scheduled_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expected_failure_message = 'Account is deactivated'
        logger = mock.Mock()
        self.create_scheduled_message()
        scheduled_message = self.last_scheduled_message()
        self.assertTrue(scheduled_message.sender.is_active)
        message_before_deactivation = most_recent_message(scheduled_message.sender)
        more_than_scheduled_delivery_datetime = scheduled_message.scheduled_timestamp + datetime.timedelta(minutes=1)
        with time_machine.travel(more_than_scheduled_delivery_datetime, tick=False):
            scheduled_message = self.last_scheduled_message()
            change_user_is_active(scheduled_message.sender, False)
            self.verify_deliver_scheduled_message_failure(scheduled_message, logger, expected_failure_message)
        self.assertFalse(scheduled_message.sender.is_active)
        message_after_deactivation = most_recent_message(scheduled_message.sender)
        self.assertEqual(message_after_deactivation.content, message_before_deactivation.content)
        self.assertNotIn(expected_failure_message, message_after_deactivation.content)

    def test_delivery_type_reminder_failed_to_deliver_scheduled_message_unknown_exception(self) -> None:
        if False:
            i = 10
            return i + 15
        logger = mock.Mock()
        self.create_scheduled_message()
        scheduled_message = self.last_scheduled_message()
        more_than_scheduled_delivery_datetime = scheduled_message.scheduled_timestamp + datetime.timedelta(minutes=1)
        with time_machine.travel(more_than_scheduled_delivery_datetime, tick=False):
            scheduled_message = self.last_scheduled_message()
            scheduled_message.delivery_type = ScheduledMessage.REMIND
            scheduled_message.save()
            result = try_deliver_one_scheduled_message(logger)
            self.assertTrue(result)
            scheduled_message.refresh_from_db()
            logger.info.assert_called_once_with('Sending scheduled message %s with date %s (sender: %s)', scheduled_message.id, scheduled_message.scheduled_timestamp, scheduled_message.sender_id)
            logger.exception.assert_called_once_with('Unexpected error sending scheduled message %s (sent: %s)', scheduled_message.id, scheduled_message.delivered, stack_info=True)
            self.assertTrue(scheduled_message.failed)
        realm = scheduled_message.realm
        msg = most_recent_message(scheduled_message.sender)
        self.assertEqual(msg.recipient.type, msg.recipient.PERSONAL)
        self.assertEqual(msg.sender_id, self.notification_bot(realm).id)
        self.assertIn('Internal server error', msg.content)

    def test_editing_failed_send_scheduled_message(self) -> None:
        if False:
            return 10
        expected_failure_message = 'Message could not be sent at the scheduled time.'
        logger = mock.Mock()
        self.create_scheduled_message()
        scheduled_message = self.last_scheduled_message()
        too_late_to_send_message_datetime = scheduled_message.scheduled_timestamp + datetime.timedelta(minutes=SCHEDULED_MESSAGE_LATE_CUTOFF_MINUTES + 1)
        with time_machine.travel(too_late_to_send_message_datetime, tick=False):
            self.verify_deliver_scheduled_message_failure(scheduled_message, logger, expected_failure_message)
            payload_without_timestamp = {'topic': 'Failed to send'}
            response = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload_without_timestamp)
            self.assert_json_error(response, 'Scheduled delivery time must be in the future.')
        new_delivery_datetime = timezone_now() + datetime.timedelta(minutes=60)
        new_delivery_timestamp = int(new_delivery_datetime.timestamp())
        scheduled_message_id = scheduled_message.id
        payload_with_timestamp = {'scheduled_delivery_timestamp': new_delivery_timestamp}
        response = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload_with_timestamp)
        self.assert_json_success(response)
        scheduled_message = self.last_scheduled_message()
        self.assertEqual(scheduled_message.id, scheduled_message_id)
        self.assertFalse(scheduled_message.failed)
        self.assertIsNone(scheduled_message.failure_message)

    def test_scheduling_in_past(self) -> None:
        if False:
            return 10
        content = 'Test message'
        verona_stream_id = self.get_stream_id('Verona')
        scheduled_delivery_timestamp = int(time.time() - 86400)
        result = self.do_schedule_message('stream', verona_stream_id, content + ' 1', scheduled_delivery_timestamp)
        self.assert_json_error(result, 'Scheduled delivery time must be in the future.')

    def test_edit_schedule_message(self) -> None:
        if False:
            i = 10
            return i + 15
        content = 'Original test message'
        scheduled_delivery_timestamp = int(time.time() + 86400)
        verona_stream_id = self.get_stream_id('Verona')
        result = self.do_schedule_message('stream', verona_stream_id, content, scheduled_delivery_timestamp)
        scheduled_message = self.last_scheduled_message()
        self.assert_json_success(result)
        self.assertEqual(scheduled_message.recipient.type, Recipient.STREAM)
        self.assertEqual(scheduled_message.content, 'Original test message')
        self.assertEqual(scheduled_message.topic_name(), 'Test topic')
        self.assertEqual(scheduled_message.scheduled_timestamp, timestamp_to_datetime(scheduled_delivery_timestamp))
        scheduled_message_id = scheduled_message.id
        payload: Dict[str, Any]
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}')
        self.assert_json_error(result, 'Nothing to change')
        payload = {'type': 'direct'}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}', payload)
        self.assert_json_error(result, 'Recipient required when updating type of scheduled message.')
        othello = self.example_user('othello')
        to = [othello.id]
        payload = {'type': 'direct', 'to': orjson.dumps(to).decode()}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}', payload)
        self.assert_json_success(result)
        scheduled_message = self.get_scheduled_message(str(scheduled_message_id))
        self.assertNotEqual(scheduled_message.recipient.type, Recipient.STREAM)
        payload = {'topic': 'Direct message topic'}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}', payload)
        self.assert_json_success(result)
        scheduled_message = self.get_scheduled_message(str(scheduled_message_id))
        self.assertEqual(scheduled_message.topic_name(), '')
        payload = {'type': 'stream', 'to': orjson.dumps(verona_stream_id).decode()}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}', payload)
        self.assert_json_error(result, 'Topic required when updating scheduled message type to stream.')
        payload = {'type': 'stream', 'to': orjson.dumps(verona_stream_id).decode(), 'topic': 'New test topic'}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}', payload)
        self.assert_json_success(result)
        scheduled_message = self.get_scheduled_message(str(scheduled_message_id))
        self.assertEqual(scheduled_message.recipient.type, Recipient.STREAM)
        self.assertEqual(scheduled_message.topic_name(), 'New test topic')
        new_scheduled_delivery_timestamp = int(time.time() - 86400)
        payload = {'scheduled_delivery_timestamp': new_scheduled_delivery_timestamp}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}', payload)
        self.assert_json_error(result, 'Scheduled delivery time must be in the future.')
        edited_content = 'Edited test message'
        new_scheduled_delivery_timestamp = scheduled_delivery_timestamp + int(time.time() + 3 * 86400)
        payload = {'content': edited_content, 'scheduled_delivery_timestamp': new_scheduled_delivery_timestamp}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message_id}', payload)
        self.assert_json_success(result)
        scheduled_message = self.get_scheduled_message(str(scheduled_message_id))
        self.assertEqual(scheduled_message.content, edited_content)
        self.assertEqual(scheduled_message.topic_name(), 'New test topic')
        self.assertEqual(scheduled_message.scheduled_timestamp, timestamp_to_datetime(new_scheduled_delivery_timestamp))
        edited_content = 'Final content edit for test'
        payload = {'topic': 'Another topic for test', 'content': edited_content}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload)
        self.assert_json_success(result)
        scheduled_message = self.get_scheduled_message(str(scheduled_message.id))
        self.assertEqual(scheduled_message.content, edited_content)
        self.assertEqual(scheduled_message.topic_name(), 'Another topic for test')
        payload = {'topic': 'Final topic for test'}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload)
        self.assert_json_success(result)
        scheduled_message = self.get_scheduled_message(str(scheduled_message.id))
        self.assertEqual(scheduled_message.recipient.type, Recipient.STREAM)
        self.assertEqual(scheduled_message.content, edited_content)
        self.assertEqual(scheduled_message.topic_name(), 'Final topic for test')
        self.assertEqual(scheduled_message.scheduled_timestamp, timestamp_to_datetime(new_scheduled_delivery_timestamp))

    def test_fetch_scheduled_messages(self) -> None:
        if False:
            return 10
        self.login('hamlet')
        result = self.client_get('/json/scheduled_messages')
        self.assert_json_success(result)
        self.assert_length(orjson.loads(result.content)['scheduled_messages'], 0)
        verona_stream_id = self.get_stream_id('Verona')
        content = 'Test message'
        scheduled_delivery_timestamp = int(time.time() + 86400)
        self.do_schedule_message('stream', verona_stream_id, content, scheduled_delivery_timestamp)
        result = self.client_get('/json/scheduled_messages')
        self.assert_json_success(result)
        scheduled_messages = orjson.loads(result.content)['scheduled_messages']
        self.assert_length(scheduled_messages, 1)
        self.assertEqual(scheduled_messages[0]['scheduled_message_id'], self.last_scheduled_message().id)
        self.assertEqual(scheduled_messages[0]['content'], content)
        self.assertEqual(scheduled_messages[0]['to'], verona_stream_id)
        self.assertEqual(scheduled_messages[0]['type'], 'stream')
        self.assertEqual(scheduled_messages[0]['topic'], 'Test topic')
        self.assertEqual(scheduled_messages[0]['scheduled_delivery_timestamp'], scheduled_delivery_timestamp)
        othello = self.example_user('othello')
        result = self.do_schedule_message('direct', [othello.id], content + ' 3', scheduled_delivery_timestamp)
        result = self.client_get('/json/scheduled_messages')
        self.assert_json_success(result)
        self.assert_length(orjson.loads(result.content)['scheduled_messages'], 2)
        self.logout()
        self.login('othello')
        result = self.client_get('/json/scheduled_messages')
        self.assert_json_success(result)
        self.assert_length(orjson.loads(result.content)['scheduled_messages'], 0)

    def test_delete_scheduled_messages(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login('hamlet')
        content = 'Test message'
        verona_stream_id = self.get_stream_id('Verona')
        scheduled_delivery_timestamp = int(time.time() + 86400)
        self.do_schedule_message('stream', verona_stream_id, content, scheduled_delivery_timestamp)
        scheduled_message = self.last_scheduled_message()
        self.logout()
        othello = self.example_user('othello')
        result = self.api_delete(othello, f'/api/v1/scheduled_messages/{scheduled_message.id}')
        self.assert_json_error(result, 'Scheduled message does not exist', 404)
        self.login('hamlet')
        result = self.client_delete(f'/json/scheduled_messages/{scheduled_message.id}')
        self.assert_json_success(result)
        result = self.client_delete(f'/json/scheduled_messages/{scheduled_message.id}')
        self.assert_json_error(result, 'Scheduled message does not exist', 404)

    def test_attachment_handling(self) -> None:
        if False:
            while True:
                i = 10
        self.login('hamlet')
        hamlet = self.example_user('hamlet')
        verona_stream_id = self.get_stream_id('Verona')
        attachment_file1 = StringIO('zulip!')
        attachment_file1.name = 'dummy_1.txt'
        result = self.client_post('/json/user_uploads', {'file': attachment_file1})
        path_id1 = re.sub('/user_uploads/', '', result.json()['uri'])
        attachment_object1 = Attachment.objects.get(path_id=path_id1)
        attachment_file2 = StringIO('zulip!')
        attachment_file2.name = 'dummy_1.txt'
        result = self.client_post('/json/user_uploads', {'file': attachment_file2})
        path_id2 = re.sub('/user_uploads/', '', result.json()['uri'])
        attachment_object2 = Attachment.objects.get(path_id=path_id2)
        content = f'Test [zulip.txt](http://{hamlet.realm.host}/user_uploads/{path_id1})'
        scheduled_delivery_timestamp = int(time.time() + 86400)
        self.do_schedule_message('stream', verona_stream_id, content, scheduled_delivery_timestamp)
        scheduled_message = self.last_scheduled_message()
        self.assertEqual(list(attachment_object1.scheduled_messages.all().values_list('id', flat=True)), [scheduled_message.id])
        self.assertEqual(scheduled_message.has_attachment, True)
        edited_content = f'Test [zulip.txt](http://{hamlet.realm.host}/user_uploads/{path_id2})'
        payload = {'content': edited_content}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload)
        scheduled_message = self.get_scheduled_message(str(scheduled_message.id))
        self.assertEqual(list(attachment_object1.scheduled_messages.all().values_list('id', flat=True)), [])
        self.assertEqual(list(attachment_object2.scheduled_messages.all().values_list('id', flat=True)), [scheduled_message.id])
        self.assertEqual(scheduled_message.has_attachment, True)
        edited_content = 'No more attachments'
        payload = {'content': edited_content}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload)
        scheduled_message = self.get_scheduled_message(str(scheduled_message.id))
        self.assertEqual(list(attachment_object1.scheduled_messages.all().values_list('id', flat=True)), [])
        self.assertEqual(list(attachment_object2.scheduled_messages.all().values_list('id', flat=True)), [])
        self.assertEqual(scheduled_message.has_attachment, False)
        edited_content = f'Attachment is back! [zulip.txt](http://{hamlet.realm.host}/user_uploads/{path_id2})'
        payload = {'content': edited_content}
        result = self.client_patch(f'/json/scheduled_messages/{scheduled_message.id}', payload)
        scheduled_message = self.get_scheduled_message(str(scheduled_message.id))
        self.assertEqual(list(attachment_object1.scheduled_messages.all().values_list('id', flat=True)), [])
        self.assertEqual(list(attachment_object2.scheduled_messages.all().values_list('id', flat=True)), [scheduled_message.id])
        self.assertEqual(scheduled_message.has_attachment, True)