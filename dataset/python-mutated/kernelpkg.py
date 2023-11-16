"""
    :synopsis: Base class for kernelpkg modules
    :platform: Linux
    :maturity: develop
    .. versionadded:: 2018.3.0
"""
try:
    from salt.exceptions import CommandExecutionError
    from tests.support.mock import MagicMock, patch
except ImportError:
    pass

class KernelPkgTestCase:
    """
    Test cases shared by all kernelpkg virtual modules
    """

    def test_active(self):
        if False:
            while True:
                i = 10
        '\n        Test - Return return the active kernel version\n        '
        self.assertEqual(self._kernelpkg.active(), self.KERNEL_LIST[0])

    def test_latest_available_no_results(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test - Return the latest available kernel version\n        '
        mock = MagicMock(return_value='')
        with patch.dict(self._kernelpkg.__salt__, {'pkg.latest_version': mock}):
            with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[0]):
                self.assertEqual(self._kernelpkg.latest_available(), self.KERNEL_LIST[-1])

    def test_latest_available_at_latest(self):
        if False:
            print('Hello World!')
        '\n        Test - Return the latest available kernel version\n        '
        mock = MagicMock(return_value=self.LATEST)
        with patch.dict(self._kernelpkg.__salt__, {'pkg.latest_version': mock}):
            with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[-1]):
                self.assertEqual(self._kernelpkg.latest_available(), self.KERNEL_LIST[-1])

    def test_latest_available_with_updates(self):
        if False:
            print('Hello World!')
        '\n        Test - Return the latest available kernel version\n        '
        mock = MagicMock(return_value=self.LATEST)
        with patch.dict(self._kernelpkg.__salt__, {'pkg.latest_version': mock}):
            with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[0]):
                self.assertEqual(self._kernelpkg.latest_available(), self.KERNEL_LIST[-1])

    def test_latest_installed_with_updates(self):
        if False:
            i = 10
            return i + 15
        '\n        Test - Return the latest installed kernel version\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[0]):
            with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                self.assertEqual(self._kernelpkg.latest_installed(), self.KERNEL_LIST[-1])

    def test_latest_installed_at_latest(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test - Return the latest installed kernel version\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[-1]):
            with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                self.assertEqual(self._kernelpkg.latest_installed(), self.KERNEL_LIST[-1])

    def test_needs_reboot_with_update(self):
        if False:
            while True:
                i = 10
        '\n        Test - Return True if a new kernel is ready to be booted\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[0]):
            with patch.object(self._kernelpkg, 'latest_installed', return_value=self.KERNEL_LIST[1]):
                self.assertTrue(self._kernelpkg.needs_reboot())

    def test_needs_reboot_at_latest(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test - Return True if a new kernel is ready to be booted\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[1]):
            with patch.object(self._kernelpkg, 'latest_installed', return_value=self.KERNEL_LIST[1]):
                self.assertFalse(self._kernelpkg.needs_reboot())

    def test_needs_reboot_order_inverted(self):
        if False:
            while True:
                i = 10
        '\n        Test - Return True if a new kernel is ready to be booted\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[1]):
            with patch.object(self._kernelpkg, 'latest_installed', return_value=self.KERNEL_LIST[0]):
                self.assertFalse(self._kernelpkg.needs_reboot())

    def test_upgrade_not_needed_with_reboot(self):
        if False:
            while True:
                i = 10
        '\n        Test - Upgrade function when no upgrade is available and reboot has been requested\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[-1]):
            with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                result = self._kernelpkg.upgrade(reboot=True)
                self.assertIn('upgrades', result)
                self.assertEqual(result['active'], self.KERNEL_LIST[-1])
                self.assertEqual(result['latest_installed'], self.KERNEL_LIST[-1])
                self.assertEqual(result['reboot_requested'], True)
                self.assertEqual(result['reboot_required'], False)
                self._kernelpkg.__salt__['system.reboot'].assert_not_called()

    def test_upgrade_not_needed_without_reboot(self):
        if False:
            while True:
                i = 10
        '\n        Test - Upgrade function when no upgrade is available and no reboot has been requested\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[-1]):
            with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                result = self._kernelpkg.upgrade(reboot=False)
                self.assertIn('upgrades', result)
                self.assertEqual(result['active'], self.KERNEL_LIST[-1])
                self.assertEqual(result['latest_installed'], self.KERNEL_LIST[-1])
                self.assertEqual(result['reboot_requested'], False)
                self.assertEqual(result['reboot_required'], False)
                self._kernelpkg.__salt__['system.reboot'].assert_not_called()

    def test_upgrade_needed_with_reboot(self):
        if False:
            while True:
                i = 10
        '\n        Test - Upgrade function when an upgrade is available and reboot has been requested\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[0]):
            with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                result = self._kernelpkg.upgrade(reboot=True)
                self.assertIn('upgrades', result)
                self.assertEqual(result['active'], self.KERNEL_LIST[0])
                self.assertEqual(result['latest_installed'], self.KERNEL_LIST[-1])
                self.assertEqual(result['reboot_requested'], True)
                self.assertEqual(result['reboot_required'], True)
                self._kernelpkg.__salt__['system.reboot'].assert_called_once()

    def test_upgrade_needed_without_reboot(self):
        if False:
            print('Hello World!')
        '\n        Test - Upgrade function when an upgrade is available and no reboot has been requested\n        '
        with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[0]):
            with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                result = self._kernelpkg.upgrade(reboot=False)
                self.assertIn('upgrades', result)
                self.assertEqual(result['active'], self.KERNEL_LIST[0])
                self.assertEqual(result['latest_installed'], self.KERNEL_LIST[-1])
                self.assertEqual(result['reboot_requested'], False)
                self.assertEqual(result['reboot_required'], True)
                self._kernelpkg.__salt__['system.reboot'].assert_not_called()

    def test_upgrade_available_true(self):
        if False:
            i = 10
            return i + 15
        '\n        Test - upgrade_available\n        '
        with patch.object(self._kernelpkg, 'latest_available', return_value=self.KERNEL_LIST[-1]):
            with patch.object(self._kernelpkg, 'latest_installed', return_value=self.KERNEL_LIST[0]):
                self.assertTrue(self._kernelpkg.upgrade_available())

    def test_upgrade_available_false(self):
        if False:
            return 10
        '\n        Test - upgrade_available\n        '
        with patch.object(self._kernelpkg, 'latest_available', return_value=self.KERNEL_LIST[-1]):
            with patch.object(self._kernelpkg, 'latest_installed', return_value=self.KERNEL_LIST[-1]):
                self.assertFalse(self._kernelpkg.upgrade_available())

    def test_upgrade_available_inverted(self):
        if False:
            print('Hello World!')
        '\n        Test - upgrade_available\n        '
        with patch.object(self._kernelpkg, 'latest_available', return_value=self.KERNEL_LIST[0]):
            with patch.object(self._kernelpkg, 'latest_installed', return_value=self.KERNEL_LIST[-1]):
                self.assertFalse(self._kernelpkg.upgrade_available())

    def test_remove_active(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test - remove kernel package\n        '
        mock = MagicMock(return_value={'retcode': 0, 'stderr': []})
        with patch.dict(self._kernelpkg.__salt__, {'cmd.run_all': mock}):
            with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[-1]):
                with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                    self.assertRaises(CommandExecutionError, self._kernelpkg.remove, release=self.KERNEL_LIST[-1])
                    self._kernelpkg.__salt__['cmd.run_all'].assert_not_called()

    def test_remove_invalid(self):
        if False:
            while True:
                i = 10
        '\n        Test - remove kernel package\n        '
        mock = MagicMock(return_value={'retcode': 0, 'stderr': []})
        with patch.dict(self._kernelpkg.__salt__, {'cmd.run_all': mock}):
            with patch.object(self._kernelpkg, 'active', return_value=self.KERNEL_LIST[-1]):
                with patch.object(self._kernelpkg, 'list_installed', return_value=self.KERNEL_LIST):
                    self.assertRaises(CommandExecutionError, self._kernelpkg.remove, release='invalid')
                    self._kernelpkg.__salt__['cmd.run_all'].assert_not_called()