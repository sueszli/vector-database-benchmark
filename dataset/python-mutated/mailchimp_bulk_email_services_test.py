"""Tests for mailchimp services."""
from __future__ import annotations
import logging
from core import feconf
from core.platform import models
from core.platform.bulk_email import mailchimp_bulk_email_services
from core.tests import test_utils
from mailchimp3 import mailchimpclient
from typing import Dict, List
secrets_services = models.Registry.import_secrets_services()

class MailchimpServicesUnitTests(test_utils.GenericTestBase):
    """Tests for mailchimp services."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.user_email_1 = 'test1@example.com'
        self.user_email_2 = 'test2@example.com'
        self.user_email_3 = 'test3@example.com'

    class MockMailchimpClass:
        """Class to mock Mailchimp class."""
        update_call_data: Dict[str, str] = {}

        class MailchimpLists:
            """Class to mock Mailchimp lists object."""

            class MailchimpMembers:
                """Class to mock Mailchimp members object."""

                class MailchimpTags:
                    """Class to mock Mailchimp tags object."""

                    def __init__(self) -> None:
                        if False:
                            while True:
                                i = 10
                        self.tag_names: List[str] = []

                    def update(self, unused_id: str, unused_hash: str, tag_data: Dict[str, List[Dict[str, str]]]) -> None:
                        if False:
                            print('Hello World!')
                        "Mocks the tag update function in mailchimp api.\n\n                        Args:\n                            unused_id: str. List Id of mailchimp list.\n                            unused_hash: str. Subscriber hash, which is an MD5\n                                hash of subscriber's email ID.\n                            tag_data: dict. A dict with the 'tags' key\n                                containing the tags to be updated for the user.\n                        "
                        self.tag_names = [tag['name'] for tag in tag_data['tags'] if tag['status'] == 'active']

                def __init__(self) -> None:
                    if False:
                        print('Hello World!')
                    self.users_data = [{'email_hash': 'aa99b351245441b8ca95d54a52d2998c', 'status': 'unsubscribed'}, {'email_hash': '43b05f394d5611c54a1a9e8e20baee21', 'status': 'subscribed'}, {'email_hash': 'incorrecthash'}]
                    self.tags = self.MailchimpTags()

                def get(self, _list_id: str, subscriber_hash: str) -> Dict[str, str]:
                    if False:
                        while True:
                            i = 10
                    "Mocks the get function of the mailchimp api.\n\n                    Args:\n                        _list_id: str. List Id of mailchimp list.\n                        subscriber_hash: str. Subscriber hash, which is an MD5\n                            hash of subscriber's email ID.\n\n                    Raises:\n                        MailchimpError. Error 404 or 401 to mock API server\n                            error.\n\n                    Returns:\n                        dict. The updated status dict for users.\n                    "
                    if not self.users_data:
                        raise mailchimpclient.MailChimpError({'status': 401, 'detail': 'Server Error'})
                    for user in self.users_data:
                        if user['email_hash'] == subscriber_hash:
                            return user
                    raise mailchimpclient.MailChimpError({'status': 404})

                def update(self, _list_id: str, subscriber_hash: str, data: Dict[str, str]) -> None:
                    if False:
                        for i in range(10):
                            print('nop')
                    "Mocks the update function of the mailchimp api. This\n                    function just sets the payload data to a private variable\n                    to test it.\n\n                    Args:\n                        _list_id: str. List Id of mailchimp list.\n                        subscriber_hash: str. Subscriber hash, which is an MD5\n                            hash of subscriber's email ID.\n                        data: dict. Payload received.\n                    "
                    for user in self.users_data:
                        if user['email_hash'] == subscriber_hash:
                            user['status'] = data['status']

                def create(self, _list_id: str, data: Dict[str, str]) -> None:
                    if False:
                        return 10
                    'Mocks the create function of the mailchimp api. This\n                    function just sets the payload data to a private variable\n                    to test it.\n\n                    Args:\n                        _list_id: str. List Id of mailchimp list.\n                        data: dict. Payload received.\n                    '
                    if data['email_address'] == 'test3@example.com':
                        self.users_data.append({'email_hash': 'fedd8b80a7a813966263853b9af72151', 'status': data['status']})
                    elif data['email_address'] == 'test4@example.com':
                        raise mailchimpclient.MailChimpError({'status': 400, 'title': 'Forgotten Email Not Subscribed'})
                    else:
                        raise mailchimpclient.MailChimpError({'status': 404, 'title': 'Invalid email', 'detail': 'Server Issue'})

                def delete_permanent(self, _list_id: str, subscriber_hash: str) -> None:
                    if False:
                        while True:
                            i = 10
                    "Mocks the delete function of the mailchimp api. This\n                    function just sets the deleted user to a private variable\n                    to test it.\n\n                    Args:\n                        _list_id: str. List Id of mailchimp list.\n                        subscriber_hash: str. Subscriber hash, which is an MD5\n                            hash of subscriber's email ID.\n                    "
                    self.users_data = [user for user in self.users_data if user['email_hash'] != subscriber_hash]

            def __init__(self) -> None:
                if False:
                    print('Hello World!')
                self.members = self.MailchimpMembers()

        def __init__(self) -> None:
            if False:
                while True:
                    i = 10
            self.lists = self.MailchimpLists()

    def test_get_subscriber_hash(self) -> None:
        if False:
            print('Hello World!')
        sample_email = 'test@example.com'
        subscriber_hash = '55502f40dc8b7c769880b10874abc9d0'
        self.assertEqual(mailchimp_bulk_email_services._get_subscriber_hash(sample_email), subscriber_hash)
        sample_email_2 = 5
        with self.assertRaisesRegex(Exception, 'Invalid type for email. Expected string, received 5'):
            mailchimp_bulk_email_services._get_subscriber_hash(sample_email_2)

    def test_function_input_validation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        mailchimp = self.MockMailchimpClass()
        swapped_mailchimp = lambda : mailchimp
        swap_mailchimp_context = self.swap(mailchimp_bulk_email_services, '_get_mailchimp_class', swapped_mailchimp)
        with swap_mailchimp_context:
            with self.assertRaisesRegex(Exception, 'Invalid Merge Fields'):
                mailchimp_bulk_email_services.add_or_update_user_status('valid@example.com', {'INVALID': 'value'}, 'Android', can_receive_email_updates=True)
            with self.assertRaisesRegex(Exception, 'Invalid tag: Invalid'):
                mailchimp_bulk_email_services.add_or_update_user_status('valid@example.com', {}, 'Invalid', can_receive_email_updates=True)

    def test_get_mailchimp_class_errors_when_api_key_is_not_available(self) -> None:
        if False:
            while True:
                i = 10
        swap_get_secret = self.swap_with_checks(secrets_services, 'get_secret', lambda _: None, expected_args=[('MAILCHIMP_API_KEY',)])
        with self.capture_logging(min_level=logging.ERROR) as logs:
            with swap_get_secret:
                self.assertIsNone(mailchimp_bulk_email_services._get_mailchimp_class())
                self.assertItemsEqual(logs, ['Mailchimp API key is not available.'])

    def test_get_mailchimp_class_errors_when_username_is_not_available(self) -> None:
        if False:
            return 10
        swap_mailchimp_username = self.swap(feconf, 'MAILCHIMP_USERNAME', None)
        swap_get_secret = self.swap_with_checks(secrets_services, 'get_secret', lambda _: 'key', expected_args=[('MAILCHIMP_API_KEY',)])
        with self.capture_logging(min_level=logging.ERROR) as logs:
            with swap_mailchimp_username, swap_get_secret:
                self.assertIsNone(mailchimp_bulk_email_services._get_mailchimp_class())
                self.assertItemsEqual(logs, ['Mailchimp username is not set.'])

    def test_add_or_update_user_status_returns_false_when_username_is_none(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        swap_get_secret = self.swap_with_checks(secrets_services, 'get_secret', lambda _: 'key', expected_args=[('MAILCHIMP_API_KEY',)])
        with swap_get_secret:
            self.assertFalse(mailchimp_bulk_email_services.add_or_update_user_status('sample_email', {}, 'Web', can_receive_email_updates=True))

    def test_permanently_delete_user_from_list_when_username_is_none(self) -> None:
        if False:
            print('Hello World!')
        swap_get_secret = self.swap_with_checks(secrets_services, 'get_secret', lambda _: 'key', expected_args=[('MAILCHIMP_API_KEY',)])
        with swap_get_secret:
            mailchimp_bulk_email_services.permanently_delete_user_from_list('sample_email')

    def test_add_or_update_mailchimp_user_status(self) -> None:
        if False:
            i = 10
            return i + 15
        mailchimp = self.MockMailchimpClass()
        swapped_mailchimp = lambda : mailchimp
        swap_mailchimp_context = self.swap(mailchimp_bulk_email_services, '_get_mailchimp_class', swapped_mailchimp)
        swap_api = self.swap(secrets_services, 'get_secret', lambda _: 'key')
        swap_username = self.swap(feconf, 'MAILCHIMP_USERNAME', 'username')
        with swap_mailchimp_context, swap_api, swap_username:
            self.assertEqual(mailchimp.lists.members.users_data[0]['status'], 'unsubscribed')
            mailchimp_bulk_email_services.add_or_update_user_status(self.user_email_1, {}, 'Web', can_receive_email_updates=True)
            self.assertEqual(mailchimp.lists.members.users_data[0]['status'], 'subscribed')
            self.assertEqual(mailchimp.lists.members.tags.tag_names, ['Web'])
            self.assertEqual(mailchimp.lists.members.users_data[1]['status'], 'subscribed')
            mailchimp_bulk_email_services.add_or_update_user_status(self.user_email_2, {}, 'Web', can_receive_email_updates=False)
            self.assertEqual(mailchimp.lists.members.users_data[1]['status'], 'unsubscribed')
            self.assertEqual(len(mailchimp.lists.members.users_data), 3)
            return_status = mailchimp_bulk_email_services.add_or_update_user_status(self.user_email_3, {}, 'Web', can_receive_email_updates=True)
            self.assertTrue(return_status)
            self.assertEqual(mailchimp.lists.members.users_data[3]['status'], 'subscribed')
            return_status = mailchimp_bulk_email_services.add_or_update_user_status('test4@example.com', {}, 'Web', can_receive_email_updates=True)
            self.assertFalse(return_status)
            mailchimp.lists.members.users_data = None
            with self.assertRaisesRegex(Exception, 'Server Error'):
                mailchimp_bulk_email_services.add_or_update_user_status(self.user_email_1, {}, 'Web', can_receive_email_updates=True)

    def test_android_merge_fields(self) -> None:
        if False:
            return 10
        mailchimp = self.MockMailchimpClass()
        swapped_mailchimp = lambda : mailchimp
        swap_mailchimp_context = self.swap(mailchimp_bulk_email_services, '_get_mailchimp_class', swapped_mailchimp)
        swap_api = self.swap(secrets_services, 'get_secret', lambda _: 'key')
        swap_username = self.swap(feconf, 'MAILCHIMP_USERNAME', 'username')
        with swap_mailchimp_context, swap_api, swap_username:
            self.assertEqual(mailchimp.lists.members.users_data[0]['status'], 'unsubscribed')
            mailchimp_bulk_email_services.add_or_update_user_status(self.user_email_1, {'NAME': 'name'}, 'Android', can_receive_email_updates=True)
            self.assertEqual(mailchimp.lists.members.users_data[0]['status'], 'subscribed')
            self.assertEqual(mailchimp.lists.members.tags.tag_names, ['Android'])

    def test_catch_or_raise_errors_when_creating_new_invalid_user(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        mailchimp = self.MockMailchimpClass()
        swapped_mailchimp = lambda : mailchimp
        swap_mailchimp_context = self.swap(mailchimp_bulk_email_services, '_get_mailchimp_class', swapped_mailchimp)
        swap_api = self.swap(secrets_services, 'get_secret', lambda _: 'key')
        swap_username = self.swap(feconf, 'MAILCHIMP_USERNAME', 'username')
        with swap_mailchimp_context, swap_api, swap_username:
            self.assertEqual(len(mailchimp.lists.members.users_data), 3)
            return_status = mailchimp_bulk_email_services.add_or_update_user_status('test4@example.com', {}, 'Web', can_receive_email_updates=True)
            self.assertFalse(return_status)
            self.assertEqual(len(mailchimp.lists.members.users_data), 3)
            with self.assertRaisesRegex(Exception, 'Server Issue'):
                mailchimp_bulk_email_services.add_or_update_user_status('test5@example.com', {}, 'Web', can_receive_email_updates=True)

    def test_permanently_delete_user(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        mailchimp = self.MockMailchimpClass()
        swapped_mailchimp = lambda : mailchimp
        swap_mailchimp_context = self.swap(mailchimp_bulk_email_services, '_get_mailchimp_class', swapped_mailchimp)
        swap_api = self.swap(secrets_services, 'get_secret', lambda _: 'key')
        swap_username = self.swap(feconf, 'MAILCHIMP_USERNAME', 'username')
        with swap_mailchimp_context, swap_api, swap_username:
            self.assertEqual(len(mailchimp.lists.members.users_data), 3)
            mailchimp_bulk_email_services.permanently_delete_user_from_list(self.user_email_1)
            self.assertEqual(len(mailchimp.lists.members.users_data), 2)
            mailchimp.lists.members.users_data = None
            with self.assertRaisesRegex(Exception, 'Server Error'):
                mailchimp_bulk_email_services.permanently_delete_user_from_list(self.user_email_1)