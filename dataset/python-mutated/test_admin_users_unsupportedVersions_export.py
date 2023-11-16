import os
import unittest
import time
from integration_tests.env_variable_names import SLACK_SDK_TEST_GRID_ORG_ADMIN_USER_TOKEN
from slack_sdk.web import WebClient

class TestWebClient(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.org_admin_token = os.environ[SLACK_SDK_TEST_GRID_ORG_ADMIN_USER_TOKEN]
        self.client: WebClient = WebClient(token=self.org_admin_token)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_no_args(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.admin_users_unsupportedVersions_export()
        self.assertIsNone(response.get('error'))

    def test_full_args(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.admin_users_unsupportedVersions_export(date_end_of_support=int(round(time.time())) + 60 * 60 * 24 * 120, date_sessions_started=0)
        self.assertIsNone(response.get('error'))

    def test_full_args_str(self):
        if False:
            while True:
                i = 10
        response = self.client.admin_users_unsupportedVersions_export(date_end_of_support=str(int(round(time.time())) + 60 * 60 * 24 * 120), date_sessions_started='0')
        self.assertIsNone(response.get('error'))