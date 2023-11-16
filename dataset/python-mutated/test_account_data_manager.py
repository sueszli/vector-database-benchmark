from twisted.test.proto_helpers import MemoryReactor
from synapse.api.errors import SynapseError
from synapse.rest import admin
from synapse.server import HomeServer
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class ModuleApiTestCase(HomeserverTestCase):
    servlets = [admin.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self._store = homeserver.get_datastores().main
        self._module_api = homeserver.get_module_api()
        self._account_data_mgr = self._module_api.account_data_manager
        self.user_id = self.register_user('kristina', 'secret')

    def test_get_global(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that getting global account data through the module API works as\n        expected, including getting `None` for unset account data.\n        '
        self.get_success(self._store.add_account_data_for_user(self.user_id, 'test.data', {'wombat': True}))
        self.assertEqual(self.get_success(self._account_data_mgr.get_global(self.user_id, 'test.data')), {'wombat': True})
        self.assertIsNone(self.get_success(self._account_data_mgr.get_global(self.user_id, 'no.data.at.all')))

    def test_get_global_validation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that invalid or remote user IDs are treated as errors and raised as exceptions,\n        whilst getting global account data for a user.\n\n        This is a design choice to try and communicate potential bugs to modules\n        earlier on.\n        '
        with self.assertRaises(SynapseError):
            self.get_success_or_raise(self._account_data_mgr.get_global("this isn't a user id", 'test.data'))
        with self.assertRaises(ValueError):
            self.get_success_or_raise(self._account_data_mgr.get_global('@valid.but:remote', 'test.data'))

    def test_get_global_no_mutability(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that modules can't introduce bugs into Synapse by mutating the result\n        of `get_global`.\n        "
        self.get_success(self._store.add_account_data_for_user(self.user_id, 'test.data', {'wombat': True}))
        the_data = self.get_success(self._account_data_mgr.get_global(self.user_id, 'test.data'))
        with self.assertRaises(TypeError):
            the_data['wombat'] = False

    def test_put_global(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Tests that written account data using `put_global` can be read out again later.\n        '
        self.get_success(self._module_api.account_data_manager.put_global(self.user_id, 'test.data', {'wombat': True}))
        self.assertEqual(self.get_success(self._store.get_global_account_data_by_type_for_user(self.user_id, 'test.data')), {'wombat': True})

    def test_put_global_validation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that a module can't write account data to user IDs that don't have\n        actual users registered to them.\n        Modules also must supply the correct types.\n        "
        with self.assertRaises(SynapseError):
            self.get_success_or_raise(self._account_data_mgr.put_global("this isn't a user id", 'test.data', {}))
        with self.assertRaises(ValueError):
            self.get_success_or_raise(self._account_data_mgr.put_global('@valid.but:remote', 'test.data', {}))
        with self.assertRaises(ValueError):
            self.get_success_or_raise(self._module_api.account_data_manager.put_global('@notregistered:test', 'test.data', {}))
        with self.assertRaises(TypeError):
            self.get_success_or_raise(self._module_api.account_data_manager.put_global(self.user_id, 42, {}))
        with self.assertRaises(TypeError):
            self.get_success_or_raise(self._module_api.account_data_manager.put_global(self.user_id, 'test.data', 42))