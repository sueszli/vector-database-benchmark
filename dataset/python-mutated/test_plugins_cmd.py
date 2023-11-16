import subprocess
import unittest
from modelscope.utils.plugins import PluginsManager

@unittest.skipUnless(False, reason='For it modify torch version')
class PluginsCMDTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        print('Testing %s.%s' % (type(self).__name__, self._testMethodName))
        self.package = 'adaseq'
        self.plugins_manager = PluginsManager()

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()

    def test_plugins_install(self):
        if False:
            while True:
                i = 10
        cmd = f'python -m modelscope.cli.cli plugin install {self.package}'
        (stat, output) = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        uninstall_args = [self.package, '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)

    def test_plugins_uninstall(self):
        if False:
            i = 10
            return i + 15
        uninstall_args = [self.package, '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)
        cmd = f'python -m modelscope.cli.cli plugin install {self.package}'
        (stat, output) = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        cmd = f'python -m modelscope.cli.cli plugin uninstall {self.package}'
        (stat, output) = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        uninstall_args = [self.package, '-y']
        self.plugins_manager.uninstall_plugins(uninstall_args)

    def test_plugins_list(self):
        if False:
            return 10
        cmd = 'python -m modelscope.cli.cli plugin list'
        (stat, output) = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
if __name__ == '__main__':
    unittest.main()