import logging
import os
import time
import unittest
from integration_tests.env_variable_names import SLACK_SDK_TEST_GRID_ORG_ADMIN_USER_TOKEN
from slack_sdk.scim import SCIMClient
from slack_sdk.scim.v1.group import Group, GroupMember
from slack_sdk.scim.v1.user import User, UserName, UserEmail

class TestSCIMClient(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.environ[SLACK_SDK_TEST_GRID_ORG_ADMIN_USER_TOKEN]
        self.client: SCIMClient = SCIMClient(token=self.bot_token)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_user_crud(self):
        if False:
            print('Hello World!')
        now = str(time.time())[:10]
        user = User(user_name=f'user_{now}', name=UserName(given_name='Kaz', family_name='Sera'), emails=[UserEmail(value=f'seratch+{now}@example.com')], schemas=['urn:scim:schemas:core:1.0'])
        creation = self.client.create_user(user)
        self.assertEqual(creation.status_code, 201)
        patch_result = self.client.patch_user(id=creation.user.id, partial_user=User(user_name=f'user_{now}_2', name=UserName(given_name='Kazuhiro', family_name='Sera')))
        self.assertEqual(patch_result.status_code, 200)
        patch_result_2 = self.client.patch_user(id=creation.user.id, partial_user={'user_name': f'user_{now}_3', 'name': {'given_name': 'Kaz', 'family_name': 'Sera'}})
        self.assertEqual(patch_result_2.status_code, 200)
        self.assertEqual(patch_result_2.user.user_name, f'user_{now}_3')
        self.assertEqual(patch_result_2.user.name.given_name, 'Kaz')
        patch_result_3 = self.client.patch_user(id=creation.user.id, partial_user={'userName': f'user_{now}_4', 'name': {'givenName': 'Kazuhiro', 'familyName': 'Sera'}})
        self.assertEqual(patch_result_3.status_code, 200)
        self.assertEqual(patch_result_3.user.user_name, f'user_{now}_4')
        self.assertEqual(patch_result_3.user.name.given_name, 'Kazuhiro')
        updated_user = creation.user
        updated_user.name = UserName(given_name='Foo', family_name='Bar')
        update_result = self.client.update_user(user=updated_user)
        self.assertEqual(update_result.status_code, 200)
        delete_result = self.client.delete_user(updated_user.id)
        self.assertEqual(delete_result.status_code, 200)

    def test_group_crud(self):
        if False:
            while True:
                i = 10
        now = str(time.time())[:10]
        user = User(user_name=f'user_{now}', name=UserName(given_name='Kaz', family_name='Sera'), emails=[UserEmail(value=f'seratch+{now}@example.com')], schemas=['urn:scim:schemas:core:1.0'])
        user_creation = self.client.create_user(user)
        group = Group(display_name=f'TestGroup_{now}', members=[GroupMember(value=user_creation.user.id)])
        creation = self.client.create_group(group)
        self.assertEqual(creation.status_code, 201)
        group = creation.group
        patch_result = self.client.patch_group(id=group.id, partial_group=Group(display_name=f'Test Group{now}_2'))
        self.assertEqual(patch_result.status_code, 204)
        updated_group = group
        updated_group.display_name = f'Test Group{now}_3'
        update_result = self.client.update_group(updated_group)
        self.assertEqual(update_result.status_code, 200)
        delete_result = self.client.delete_group(updated_group.id)
        self.assertEqual(delete_result.status_code, 204)