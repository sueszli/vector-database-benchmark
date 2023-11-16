"""
integration tests for mac_ports
"""
import pytest
from tests.support.case import ModuleCase

@pytest.mark.skip_if_not_root
@pytest.mark.skip_if_binaries_missing('port')
@pytest.mark.skip_unless_on_darwin
class MacPortsModuleTest(ModuleCase):
    """
    Validate the mac_ports module
    """
    AGREE_INSTALLED = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current settings\n        '
        self.AGREE_INSTALLED = 'agree' in self.run_function('pkg.list_pkgs')
        self.run_function('pkg.refresh_db')

    def tearDown(self):
        if False:
            return 10
        '\n        Reset to original settings\n        '
        if not self.AGREE_INSTALLED:
            self.run_function('pkg.remove', ['agree'])

    @pytest.mark.destructive_test
    def test_list_pkgs(self):
        if False:
            while True:
                i = 10
        '\n        Test pkg.list_pkgs\n        '
        self.run_function('pkg.install', ['agree'])
        self.assertIsInstance(self.run_function('pkg.list_pkgs'), dict)
        self.assertIn('agree', self.run_function('pkg.list_pkgs'))

    @pytest.mark.destructive_test
    def test_latest_version(self):
        if False:
            return 10
        '\n        Test pkg.latest_version\n        '
        self.run_function('pkg.install', ['agree'])
        result = self.run_function('pkg.latest_version', ['agree'], refresh=False)
        self.assertIsInstance(result, dict)
        self.assertIn('agree', result)

    @pytest.mark.destructive_test
    def test_remove(self):
        if False:
            i = 10
            return i + 15
        '\n        Test pkg.remove\n        '
        self.run_function('pkg.install', ['agree'])
        removed = self.run_function('pkg.remove', ['agree'])
        self.assertIsInstance(removed, dict)
        self.assertIn('agree', removed)

    @pytest.mark.destructive_test
    def test_install(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test pkg.install\n        '
        self.run_function('pkg.remove', ['agree'])
        installed = self.run_function('pkg.install', ['agree'])
        self.assertIsInstance(installed, dict)
        self.assertIn('agree', installed)

    def test_list_upgrades(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test pkg.list_upgrades\n        '
        self.assertIsInstance(self.run_function('pkg.list_upgrades', refresh=False), dict)

    @pytest.mark.destructive_test
    def test_upgrade_available(self):
        if False:
            print('Hello World!')
        '\n        Test pkg.upgrade_available\n        '
        self.run_function('pkg.install', ['agree'])
        self.assertFalse(self.run_function('pkg.upgrade_available', ['agree'], refresh=False))

    def test_refresh_db(self):
        if False:
            i = 10
            return i + 15
        '\n        Test pkg.refresh_db\n        '
        self.assertTrue(self.run_function('pkg.refresh_db'))

    @pytest.mark.destructive_test
    def test_upgrade(self):
        if False:
            while True:
                i = 10
        '\n        Test pkg.upgrade\n        '
        results = self.run_function('pkg.upgrade', refresh=False)
        self.assertIsInstance(results, dict)
        self.assertTrue(results['result'])