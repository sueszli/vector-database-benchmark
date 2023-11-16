import pytest
from tests.support.case import ModuleCase

@pytest.mark.skip_unless_on_windows
@pytest.mark.windows_whitelisted
class FirewallTest(ModuleCase):
    """
    Validate windows firewall module
    """

    def _pre_firewall_status(self, pre_run):
        if False:
            i = 10
            return i + 15
        post_run = self.run_function('firewall.get_config')
        network = ['Domain', 'Public', 'Private']
        for net in network:
            if post_run[net] != pre_run[net]:
                if pre_run[net]:
                    self.assertTrue(self.run_function('firewall.enable', profile=net))
                else:
                    self.assertTrue(self.run_function('firewall.disable', profile=net))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_firewall_get_config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test firewall.get_config\n        '
        pre_run = self.run_function('firewall.get_config')
        self.assertTrue(self.run_function('firewall.enable', profile='allprofiles'))
        ret = self.run_function('firewall.get_config')
        network = ['Domain', 'Public', 'Private']
        for net in network:
            self.assertTrue(ret[net])
        self._pre_firewall_status(pre_run)

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_firewall_disable(self):
        if False:
            while True:
                i = 10
        '\n        test firewall.disable\n        '
        pre_run = self.run_function('firewall.get_config')
        network = 'Private'
        ret = self.run_function('firewall.get_config')[network]
        if not ret:
            self.assertTrue(self.run_function('firewall.enable', profile=network))
        self.assertTrue(self.run_function('firewall.disable', profile=network))
        ret = self.run_function('firewall.get_config')[network]
        self.assertFalse(ret)
        self._pre_firewall_status(pre_run)

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_firewall_enable(self):
        if False:
            i = 10
            return i + 15
        '\n        test firewall.enable\n        '
        pre_run = self.run_function('firewall.get_config')
        network = 'Private'
        ret = self.run_function('firewall.get_config')[network]
        if ret:
            self.assertTrue(self.run_function('firewall.disable', profile=network))
        self.assertTrue(self.run_function('firewall.enable', profile=network))
        ret = self.run_function('firewall.get_config')[network]
        self.assertTrue(ret)
        self._pre_firewall_status(pre_run)

    @pytest.mark.slow_test
    def test_firewall_get_rule(self):
        if False:
            return 10
        '\n        test firewall.get_rule\n        '
        rule = 'Remote Event Log Management (NP-In)'
        ret = self.run_function('firewall.get_rule', [rule])
        checks = ['Private', 'LocalPort', 'RemotePort']
        for check in checks:
            self.assertIn(check, ret[rule])

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_firewall_add_delete_rule(self):
        if False:
            return 10
        '\n        test firewall.add_rule and delete_rule\n        '
        rule = 'test rule'
        port = '8080'
        add_rule = self.run_function('firewall.add_rule', [rule, port])
        ret = self.run_function('firewall.get_rule', [rule])
        self.assertIn(rule, ret[rule])
        self.assertIn(port, ret[rule])
        self.assertTrue(self.run_function('firewall.delete_rule', [rule, port]))
        ret = self.run_function('firewall.get_rule', [rule])
        self.assertNotIn(rule, ret)
        self.assertNotIn(port, ret)
        self.assertIn('No rules match the specified criteria.', ret)