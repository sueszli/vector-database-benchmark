from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import AccountDataTypes
from synapse.push.rulekinds import PRIORITY_CLASS_MAP
from synapse.rest import admin
from synapse.rest.client import account, login
from synapse.server import HomeServer
from synapse.synapse_rust.push import PushRule
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class DeactivateAccountTestCase(HomeserverTestCase):
    servlets = [login.register_servlets, admin.register_servlets, account.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self._store = hs.get_datastores().main
        self.user = self.register_user('user', 'pass')
        self.token = self.login('user', 'pass')

    def _deactivate_my_account(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Deactivates the account `self.user` using `self.token` and asserts\n        that it returns a 200 success code.\n        '
        req = self.make_request('POST', 'account/deactivate', {'auth': {'type': 'm.login.password', 'user': self.user, 'password': 'pass'}, 'erase': True}, access_token=self.token)
        self.assertEqual(req.code, 200, req)

    def test_global_account_data_deleted_upon_deactivation(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Tests that global account data is removed upon deactivation.\n        '
        self.get_success(self._store.add_account_data_for_user(self.user, AccountDataTypes.DIRECT, {'@someone:remote': ['!somewhere:remote']}))
        self.assertIsNotNone(self.get_success(self._store.get_global_account_data_by_type_for_user(self.user, AccountDataTypes.DIRECT)))
        self._deactivate_my_account()
        self.assertIsNone(self.get_success(self._store.get_global_account_data_by_type_for_user(self.user, AccountDataTypes.DIRECT)))

    def test_room_account_data_deleted_upon_deactivation(self) -> None:
        if False:
            return 10
        '\n        Tests that room account data is removed upon deactivation.\n        '
        room_id = '!room:test'
        self.get_success(self._store.add_account_data_to_room(self.user, room_id, 'm.fully_read', {'event_id': '$aaaa:test'}))
        self.assertIsNotNone(self.get_success(self._store.get_account_data_for_room_and_type(self.user, room_id, 'm.fully_read')))
        self._deactivate_my_account()
        self.assertIsNone(self.get_success(self._store.get_account_data_for_room_and_type(self.user, room_id, 'm.fully_read')))

    def _is_custom_rule(self, push_rule: PushRule) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Default rules start with a dot: such as .m.rule and .im.vector.\n        This function returns true iff a rule is custom (not default).\n        '
        return '/.' not in push_rule.rule_id

    def test_push_rules_deleted_upon_account_deactivation(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Push rules are a special case of account data.\n        They are stored separately but get sent to the client as account data in /sync.\n        This tests that deactivating a user deletes push rules along with the rest\n        of their account data.\n        '
        self.get_success(self._store.add_push_rule(self.user, 'personal.override.rule1', PRIORITY_CLASS_MAP['override'], [], []))
        filtered_push_rules = self.get_success(self._store.get_push_rules_for_user(self.user))
        push_rules = [r for (r, _) in filtered_push_rules.rules() if self._is_custom_rule(r)]
        self.assertEqual(len(push_rules), 1)
        self.assertEqual(push_rules[0].rule_id, 'personal.override.rule1')
        self.assertEqual(push_rules[0].priority_class, 5)
        self.assertEqual(push_rules[0].conditions, [])
        self.assertEqual(push_rules[0].actions, [])
        self._deactivate_my_account()
        filtered_push_rules = self.get_success(self._store.get_push_rules_for_user(self.user))
        push_rules = [r for (r, _) in filtered_push_rules.rules() if self._is_custom_rule(r)]
        self.assertEqual(push_rules, [], push_rules)

    def test_ignored_users_deleted_upon_deactivation(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Ignored users are a special case of account data.\n        They get denormalised into the `ignored_users` table upon being stored as\n        account data.\n        Test that a user's list of ignored users is deleted upon deactivation.\n        "
        self.get_success(self._store.add_account_data_for_user(self.user, AccountDataTypes.IGNORED_USER_LIST, {'ignored_users': {'@sheltie:test': {}}}))
        self.assertEqual(self.get_success(self._store.ignored_by('@sheltie:test')), {self.user})
        self._deactivate_my_account()
        self.assertEqual(self.get_success(self._store.ignored_by('@sheltie:test')), set())

    def _rerun_retroactive_account_data_deletion_update(self) -> None:
        if False:
            i = 10
            return i + 15
        self._store.db_pool.updates._all_done = False
        self.get_success(self._store.db_pool.simple_insert('background_updates', {'update_name': 'delete_account_data_for_deactivated_users', 'progress_json': '{}'}))
        self.wait_for_background_updates()

    def test_account_data_deleted_retroactively_by_background_update_if_deactivated(self) -> None:
        if False:
            print('Hello World!')
        '\n        Tests that a user, who deactivated their account before account data was\n        deleted automatically upon deactivation, has their account data retroactively\n        scrubbed by the background update.\n        '
        self._deactivate_my_account()
        self.get_success(self._store.add_account_data_for_user(self.user, AccountDataTypes.DIRECT, {'@someone:remote': ['!somewhere:remote']}))
        self.assertIsNotNone(self.get_success(self._store.get_global_account_data_by_type_for_user(self.user, AccountDataTypes.DIRECT)))
        self._rerun_retroactive_account_data_deletion_update()
        self.assertIsNone(self.get_success(self._store.get_global_account_data_by_type_for_user(self.user, AccountDataTypes.DIRECT)))

    def test_account_data_preserved_by_background_update_if_not_deactivated(self) -> None:
        if False:
            return 10
        '\n        Tests that the background update does not scrub account data for users that have\n        not been deactivated.\n        '
        self.get_success(self._store.add_account_data_for_user(self.user, AccountDataTypes.DIRECT, {'@someone:remote': ['!somewhere:remote']}))
        self.assertIsNotNone(self.get_success(self._store.get_global_account_data_by_type_for_user(self.user, AccountDataTypes.DIRECT)))
        self._rerun_retroactive_account_data_deletion_update()
        self.assertIsNotNone(self.get_success(self._store.get_global_account_data_by_type_for_user(self.user, AccountDataTypes.DIRECT)))

    def test_deactivate_account_needs_auth(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests that making a request to /deactivate with an empty body\n        succeeds in starting the user-interactive auth flow.\n        '
        req = self.make_request('POST', 'account/deactivate', {}, access_token=self.token)
        self.assertEqual(req.code, 401, req)
        self.assertEqual(req.json_body['flows'], [{'stages': ['m.login.password']}])