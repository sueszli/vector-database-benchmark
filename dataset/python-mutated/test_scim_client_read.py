import logging
import os
import unittest
from integration_tests.env_variable_names import SLACK_SDK_TEST_GRID_ORG_ADMIN_USER_TOKEN
from slack_sdk.scim import SCIMClient, SCIMResponse

class TestSCIMClient(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.environ[SLACK_SDK_TEST_GRID_ORG_ADMIN_USER_TOKEN]
        self.client: SCIMClient = SCIMClient(token=self.bot_token)

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    def test_api_call(self):
        if False:
            while True:
                i = 10
        response: SCIMResponse = self.client.api_call(http_verb='GET', path='Users', query_params={'startIndex': 1, 'count': 1})
        self.assertIsNotNone(response)
        self.logger.info(response.snake_cased_body)
        self.assertEqual(response.snake_cased_body['start_index'], 1)
        self.assertIsNotNone(response.snake_cased_body['resources'][0]['id'])

    def test_lookup_users(self):
        if False:
            while True:
                i = 10
        search_result = self.client.search_users(start_index=1, count=1)
        self.assertIsNotNone(search_result)
        self.logger.info(search_result.snake_cased_body)
        self.assertEqual(search_result.snake_cased_body['start_index'], 1)
        self.assertIsNotNone(search_result.snake_cased_body['resources'][0]['id'])
        self.assertEqual(search_result.users[0].id, search_result.snake_cased_body['resources'][0]['id'])
        read_result = self.client.read_user(search_result.users[0].id)
        self.assertIsNotNone(read_result)
        self.logger.info(read_result.snake_cased_body)
        self.assertEqual(read_result.user.id, search_result.users[0].id)

    def test_lookup_users_error(self):
        if False:
            return 10
        error_result = self.client.search_users(start_index=1, count=1, filter='foo')
        self.assertEqual(error_result.errors.code, 400)
        self.assertEqual(error_result.errors.description, 'no_filters (is_aggregate_call=1)')

    def test_lookup_groups(self):
        if False:
            return 10
        search_result = self.client.search_groups(start_index=1, count=1)
        self.assertIsNotNone(search_result)
        self.logger.info(search_result.snake_cased_body)
        self.assertEqual(search_result.snake_cased_body['start_index'], 1)
        self.assertIsNotNone(search_result.snake_cased_body['resources'][0]['id'])
        self.assertEqual(search_result.groups[0].id, search_result.snake_cased_body['resources'][0]['id'])
        read_result = self.client.read_group(search_result.groups[0].id)
        self.assertIsNotNone(read_result)
        self.logger.info(read_result.snake_cased_body)
        self.assertEqual(read_result.group.id, search_result.groups[0].id)

    def test_lookup_groups_error(self):
        if False:
            i = 10
            return i + 15
        error_result = self.client.search_groups(start_index=1, count=-1, filter='foo')
        self.assertEqual(error_result.errors.code, 400)
        self.assertEqual(error_result.errors.description, 'no_filters (is_aggregate_call=1)')