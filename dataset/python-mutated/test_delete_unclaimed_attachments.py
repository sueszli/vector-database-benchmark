import os
import re
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional
import time_machine
from django.conf import settings
from django.utils.timezone import now as timezone_now
from zerver.actions.message_delete import do_delete_messages
from zerver.actions.scheduled_messages import check_schedule_message, delete_scheduled_message
from zerver.actions.uploads import do_delete_old_unclaimed_attachments
from zerver.lib.retention import clean_archived_data
from zerver.lib.test_classes import UploadSerializeMixin, ZulipTestCase
from zerver.models import ArchivedAttachment, Attachment, Message, UserProfile, get_client

class UnclaimedAttachmentTest(UploadSerializeMixin, ZulipTestCase):

    def make_attachment(self, filename: str, when: Optional[datetime]=None, uploader: Optional[UserProfile]=None) -> Attachment:
        if False:
            return 10
        if when is None:
            when = timezone_now() - timedelta(weeks=2)
        if uploader is None:
            uploader = self.example_user('hamlet')
        self.login_user(uploader)
        with time_machine.travel(when, tick=False):
            file_obj = StringIO('zulip!')
            file_obj.name = filename
            response = self.assert_json_success(self.client_post('/json/user_uploads', {'file': file_obj}))
            path_id = re.sub('/user_uploads/', '', response['uri'])
            return Attachment.objects.get(path_id=path_id)

    def assert_exists(self, attachment: Attachment, *, has_file: bool, has_attachment: bool, has_archived_attachment: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert settings.LOCAL_FILES_DIR
        self.assertEqual(os.path.isfile(os.path.join(settings.LOCAL_FILES_DIR, attachment.path_id)), has_file)
        self.assertEqual(Attachment.objects.filter(id=attachment.id).exists(), has_attachment)
        self.assertEqual(ArchivedAttachment.objects.filter(id=attachment.id).exists(), has_archived_attachment)

    def test_delete_unused_upload(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        unused_attachment = self.make_attachment('unused.txt')
        self.assert_exists(unused_attachment, has_file=True, has_attachment=True, has_archived_attachment=False)
        do_delete_old_unclaimed_attachments(3)
        self.assert_exists(unused_attachment, has_file=True, has_attachment=True, has_archived_attachment=False)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(unused_attachment, has_file=False, has_attachment=False, has_archived_attachment=False)

    def test_delete_used_upload(self) -> None:
        if False:
            return 10
        hamlet = self.example_user('hamlet')
        attachment = self.make_attachment('used.txt')
        self.subscribe(hamlet, 'Denmark')
        body = f'Some files here ...[zulip.txt](http://{hamlet.realm.host}/user_uploads/{attachment.path_id})'
        self.send_stream_message(hamlet, 'Denmark', body, 'test')
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=False)

    def test_delete_upload_archived_message(self) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        attachment = self.make_attachment('used.txt')
        self.subscribe(hamlet, 'Denmark')
        body = f'Some files here ...[zulip.txt](http://{hamlet.realm.host}/user_uploads/{attachment.path_id})'
        message_id = self.send_stream_message(hamlet, 'Denmark', body, 'test')
        do_delete_messages(hamlet.realm, [Message.objects.get(id=message_id)])
        self.assert_exists(attachment, has_file=True, has_attachment=False, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=True, has_attachment=False, has_archived_attachment=True)
        with self.settings(ARCHIVED_DATA_VACUUMING_DELAY_DAYS=0):
            clean_archived_data()
        self.assert_exists(attachment, has_file=True, has_attachment=False, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=False, has_attachment=False, has_archived_attachment=False)

    def test_delete_one_message(self) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        attachment = self.make_attachment('used.txt')
        self.subscribe(hamlet, 'Denmark')
        body = f'Some files here ...[zulip.txt](http://{hamlet.realm.host}/user_uploads/{attachment.path_id})'
        first_message_id = self.send_stream_message(hamlet, 'Denmark', body, 'test')
        second_message_id = self.send_stream_message(hamlet, 'Denmark', body, 'test')
        do_delete_messages(hamlet.realm, [Message.objects.get(id=first_message_id)])
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        with self.settings(ARCHIVED_DATA_VACUUMING_DELAY_DAYS=0):
            clean_archived_data()
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        do_delete_messages(hamlet.realm, [Message.objects.get(id=second_message_id)])
        self.assert_exists(attachment, has_file=True, has_attachment=False, has_archived_attachment=True)
        with self.settings(ARCHIVED_DATA_VACUUMING_DELAY_DAYS=0):
            clean_archived_data()
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=False, has_attachment=False, has_archived_attachment=False)

    def test_delete_with_scheduled_messages(self) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        attachment = self.make_attachment('used.txt')
        self.subscribe(hamlet, 'Denmark')
        body = f'Some files here ...[zulip.txt](http://{hamlet.realm.host}/user_uploads/{attachment.path_id})'
        scheduled_message_id = check_schedule_message(hamlet, get_client('website'), 'stream', [self.get_stream_id('Denmark')], 'Test topic', body, timezone_now() + timedelta(days=365), hamlet.realm)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=False)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=False)
        delete_scheduled_message(hamlet, scheduled_message_id)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=False)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=False, has_attachment=False, has_archived_attachment=False)

    def test_delete_with_scheduled_message_and_archive(self) -> None:
        if False:
            return 10
        hamlet = self.example_user('hamlet')
        attachment = self.make_attachment('used.txt')
        self.subscribe(hamlet, 'Denmark')
        body = f'Some files here ...[zulip.txt](http://{hamlet.realm.host}/user_uploads/{attachment.path_id})'
        scheduled_message_id = check_schedule_message(hamlet, get_client('website'), 'stream', [self.get_stream_id('Denmark')], 'Test topic', body, timezone_now() + timedelta(days=365), hamlet.realm)
        sent_message_id = self.send_stream_message(hamlet, 'Denmark', body, 'test')
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=False)
        do_delete_messages(hamlet.realm, [Message.objects.get(id=sent_message_id)])
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        with self.settings(ARCHIVED_DATA_VACUUMING_DELAY_DAYS=0):
            clean_archived_data()
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        delete_scheduled_message(hamlet, scheduled_message_id)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=False, has_attachment=False, has_archived_attachment=False)

    def test_delete_with_unscheduled_message_and_archive(self) -> None:
        if False:
            return 10
        hamlet = self.example_user('hamlet')
        attachment = self.make_attachment('used.txt')
        self.subscribe(hamlet, 'Denmark')
        body = f'Some files here ...[zulip.txt](http://{hamlet.realm.host}/user_uploads/{attachment.path_id})'
        scheduled_message_id = check_schedule_message(hamlet, get_client('website'), 'stream', [self.get_stream_id('Denmark')], 'Test topic', body, timezone_now() + timedelta(days=365), hamlet.realm)
        sent_message_id = self.send_stream_message(hamlet, 'Denmark', body, 'test')
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=False)
        do_delete_messages(hamlet.realm, [Message.objects.get(id=sent_message_id)])
        delete_scheduled_message(hamlet, scheduled_message_id)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        with self.settings(ARCHIVED_DATA_VACUUMING_DELAY_DAYS=0):
            clean_archived_data()
        self.assert_exists(attachment, has_file=True, has_attachment=True, has_archived_attachment=True)
        do_delete_old_unclaimed_attachments(1)
        self.assert_exists(attachment, has_file=False, has_attachment=False, has_archived_attachment=False)