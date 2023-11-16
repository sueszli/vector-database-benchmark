from typing import Any
from unittest import mock
from typing_extensions import override
from zerver.lib.attachments import user_attachments
from zerver.lib.test_classes import ZulipTestCase
from zerver.models import Attachment

class AttachmentsTests(ZulipTestCase):

    @override
    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        user_profile = self.example_user('cordelia')
        self.attachment = Attachment.objects.create(file_name='test.txt', path_id='foo/bar/test.txt', owner=user_profile, realm=user_profile.realm, size=512)

    def test_list_by_user(self) -> None:
        if False:
            print('Hello World!')
        user_profile = self.example_user('cordelia')
        self.login_user(user_profile)
        result = self.client_get('/json/attachments')
        response_dict = self.assert_json_success(result)
        attachments = user_attachments(user_profile)
        self.assertEqual(response_dict['attachments'], attachments)

    def test_remove_attachment_exception(self) -> None:
        if False:
            while True:
                i = 10
        user_profile = self.example_user('cordelia')
        self.login_user(user_profile)
        with mock.patch('zerver.lib.attachments.delete_message_attachment', side_effect=Exception()):
            result = self.client_delete(f'/json/attachments/{self.attachment.id}')
        self.assert_json_error(result, 'An error occurred while deleting the attachment. Please try again later.')

    @mock.patch('zerver.lib.attachments.delete_message_attachment')
    def test_remove_attachment(self, ignored: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        user_profile = self.example_user('cordelia')
        self.login_user(user_profile)
        result = self.client_delete(f'/json/attachments/{self.attachment.id}')
        self.assert_json_success(result)
        attachments = user_attachments(user_profile)
        self.assertEqual(attachments, [])

    def test_list_another_user(self) -> None:
        if False:
            i = 10
            return i + 15
        user_profile = self.example_user('iago')
        self.login_user(user_profile)
        result = self.client_get('/json/attachments')
        response_dict = self.assert_json_success(result)
        self.assertEqual(response_dict['attachments'], [])

    def test_remove_another_user(self) -> None:
        if False:
            while True:
                i = 10
        user_profile = self.example_user('iago')
        self.login_user(user_profile)
        result = self.client_delete(f'/json/attachments/{self.attachment.id}')
        self.assert_json_error(result, 'Invalid attachment')
        user_profile_to_remove = self.example_user('cordelia')
        attachments = user_attachments(user_profile_to_remove)
        self.assertEqual(attachments, [self.attachment.to_dict()])

    def test_list_unauthenticated(self) -> None:
        if False:
            return 10
        result = self.client_get('/json/attachments')
        self.assert_json_error(result, 'Not logged in: API authentication or user session required', status_code=401)

    def test_delete_unauthenticated(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        result = self.client_delete(f'/json/attachments/{self.attachment.id}')
        self.assert_json_error(result, 'Not logged in: API authentication or user session required', status_code=401)