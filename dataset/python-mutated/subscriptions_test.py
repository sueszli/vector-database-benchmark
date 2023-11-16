"""Tests for user subscriptions."""
from __future__ import annotations
from core import feconf
from core.domain import subscription_services
from core.tests import test_utils
from typing import Final

class SubscriptionTests(test_utils.GenericTestBase):
    USER_EMAIL: Final = 'user@example.com'
    USER_USERNAME: Final = 'user'
    USER2_EMAIL: Final = 'user2@example.com'
    USER2_USERNAME: Final = 'user2'

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.signup(self.EDITOR_EMAIL, self.EDITOR_USERNAME)
        self.editor_id = self.get_user_id_from_email(self.EDITOR_EMAIL)
        self.signup(self.USER_EMAIL, self.USER_USERNAME)
        self.user_id = self.get_user_id_from_email(self.USER_EMAIL)
        self.signup(self.USER2_EMAIL, self.USER2_USERNAME)
        self.user_id_2 = self.get_user_id_from_email(self.USER2_EMAIL)

    def test_cannot_subscribe_without_login(self) -> None:
        if False:
            return 10
        csrf_token = self.get_new_csrf_token()
        payload = {'creator_username': self.EDITOR_USERNAME}
        response = self.post_json(feconf.SUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token, expected_status_int=401)
        self.assertEqual(response['error'], 'You do not have credentials to manage subscriptions.')

    def test_invalid_creator_username_raises_error_while_subscribing(self) -> None:
        if False:
            return 10
        self.login(self.USER_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'creator_username': 'invalid'}
        response = self.post_json(feconf.SUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token, expected_status_int=500)
        self.assertEqual(response['error'], 'No user_id found for the given username: invalid')

    def test_invalid_creator_username_raises_error_while_unsubscribing(self) -> None:
        if False:
            return 10
        self.login(self.USER_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'creator_username': 'invalid'}
        response = self.post_json(feconf.UNSUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token, expected_status_int=500)
        self.assertEqual(response['error'], 'No creator user_id found for the given creator username: invalid')

    def test_subscribe_handler(self) -> None:
        if False:
            print('Hello World!')
        'Test handler for new subscriptions to creators.'
        self.login(self.USER_EMAIL)
        csrf_token = self.get_new_csrf_token()
        payload = {'creator_username': self.EDITOR_USERNAME}
        self.post_json(feconf.SUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.editor_id), [self.user_id])
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(self.user_id), [self.editor_id])
        self.post_json(feconf.SUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.editor_id), [self.user_id])
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(self.user_id), [self.editor_id])
        self.logout()
        self.login(self.USER2_EMAIL)
        csrf_token = self.get_new_csrf_token()
        self.post_json(feconf.SUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.editor_id), [self.user_id, self.user_id_2])
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(self.user_id_2), [self.editor_id])
        self.logout()

    def test_unsubscribe_handler(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test handler for unsubscriptions.'
        payload = {'creator_username': self.EDITOR_USERNAME}
        self.login(self.USER_EMAIL)
        csrf_token = self.get_new_csrf_token()
        self.post_json(feconf.SUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.logout()
        self.login(self.USER2_EMAIL)
        csrf_token = self.get_new_csrf_token()
        self.post_json(feconf.SUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.post_json(feconf.UNSUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.editor_id), [self.user_id])
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(self.user_id_2), [])
        self.post_json(feconf.UNSUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.editor_id), [self.user_id])
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(self.user_id_2), [])
        self.logout()
        self.login(self.USER_EMAIL)
        csrf_token = self.get_new_csrf_token()
        self.post_json(feconf.UNSUBSCRIBE_URL_PREFIX, payload, csrf_token=csrf_token)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.editor_id), [])
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(self.user_id), [])