from typing import Iterable, Optional, Set
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import AccountDataTypes
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest

class IgnoredUsersTestCase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.store = self.hs.get_datastores().main
        self.user = '@user:test'

    def _update_ignore_list(self, *ignored_user_ids: Iterable[str], ignorer_user_id: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Update the account data to block the given users.'
        if ignorer_user_id is None:
            ignorer_user_id = self.user
        self.get_success(self.store.add_account_data_for_user(ignorer_user_id, AccountDataTypes.IGNORED_USER_LIST, {'ignored_users': {u: {} for u in ignored_user_ids}}))

    def assert_ignorers(self, ignored_user_id: str, expected_ignorer_user_ids: Set[str]) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self.get_success(self.store.ignored_by(ignored_user_id)), expected_ignorer_user_ids)

    def assert_ignored(self, ignorer_user_id: str, expected_ignored_user_ids: Set[str]) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self.get_success(self.store.ignored_users(ignorer_user_id)), expected_ignored_user_ids)

    def test_ignoring_users(self) -> None:
        if False:
            return 10
        'Basic adding/removing of users from the ignore list.'
        self._update_ignore_list('@other:test', '@another:remote')
        self.assert_ignored(self.user, {'@other:test', '@another:remote'})
        self.assert_ignorers('@user:test', set())
        self.assert_ignorers('@other:test', {self.user})
        self.assert_ignorers('@another:remote', {self.user})
        self._update_ignore_list('@foo:test', '@another:remote')
        self.assert_ignored(self.user, {'@foo:test', '@another:remote'})
        self.assert_ignorers('@other:test', set())
        self.assert_ignorers('@foo:test', {self.user})
        self.assert_ignorers('@another:remote', {self.user})

    def test_caching(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensure that caching works properly between different users.'
        self._update_ignore_list('@other:test')
        self.assert_ignored(self.user, {'@other:test'})
        self.assert_ignorers('@other:test', {self.user})
        self._update_ignore_list('@other:test', ignorer_user_id='@second:test')
        self.assert_ignored('@second:test', {'@other:test'})
        self.assert_ignorers('@other:test', {self.user, '@second:test'})
        self._update_ignore_list()
        self.assert_ignored(self.user, set())
        self.assert_ignorers('@other:test', {'@second:test'})

    def test_invalid_data(self) -> None:
        if False:
            print('Hello World!')
        'Invalid data ends up clearing out the ignored users list.'
        self._update_ignore_list('@other:test')
        self.assert_ignored(self.user, {'@other:test'})
        self.assert_ignorers('@other:test', {self.user})
        self.get_success(self.store.add_account_data_for_user(self.user, AccountDataTypes.IGNORED_USER_LIST, {}))
        self.assert_ignored(self.user, set())
        self.assert_ignorers('@other:test', set())
        self._update_ignore_list('@other:test')
        self.assert_ignored(self.user, {'@other:test'})
        self.assert_ignorers('@other:test', {self.user})
        self.get_success(self.store.add_account_data_for_user(self.user, AccountDataTypes.IGNORED_USER_LIST, {'ignored_users': 'unexpected'}))
        self.assert_ignored(self.user, set())
        self.assert_ignorers('@other:test', set())

    def test_ignoring_users_with_latest_stream_ids(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that ignoring users updates the latest stream ID for the ignored\n        user list account data.'

        def get_latest_ignore_streampos(user_id: str) -> Optional[int]:
            if False:
                i = 10
                return i + 15
            return self.get_success(self.store.get_latest_stream_id_for_global_account_data_by_type_for_user(user_id, AccountDataTypes.IGNORED_USER_LIST))
        self.assertIsNone(get_latest_ignore_streampos('@user:test'))
        self._update_ignore_list('@other:test', '@another:remote')
        self.assertEqual(get_latest_ignore_streampos('@user:test'), 2)
        self._update_ignore_list('@foo:test', '@another:remote')
        self.assertEqual(get_latest_ignore_streampos('@user:test'), 3)