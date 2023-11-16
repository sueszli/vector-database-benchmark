"""Tests for cli_config."""
import json
import os
import tempfile
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest

class CLIConfigTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._tmp_dir = tempfile.mkdtemp()
        self._tmp_config_path = os.path.join(self._tmp_dir, '.tfdbg_config')
        self.assertFalse(gfile.Exists(self._tmp_config_path))
        super(CLIConfigTest, self).setUp()

    def tearDown(self):
        if False:
            while True:
                i = 10
        file_io.delete_recursively(self._tmp_dir)
        super(CLIConfigTest, self).tearDown()

    def testConstructCLIConfigWithoutFile(self):
        if False:
            for i in range(10):
                print('nop')
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        self.assertEqual(20, config.get('graph_recursion_depth'))
        self.assertEqual(True, config.get('mouse_mode'))
        with self.assertRaises(KeyError):
            config.get('property_that_should_not_exist')
        self.assertTrue(gfile.Exists(self._tmp_config_path))

    def testCLIConfigForwardCompatibilityTest(self):
        if False:
            i = 10
            return i + 15
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        with open(self._tmp_config_path, 'rt') as f:
            config_json = json.load(f)
        del config_json['graph_recursion_depth']
        with open(self._tmp_config_path, 'wt') as f:
            json.dump(config_json, f)
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        self.assertEqual(20, config.get('graph_recursion_depth'))

    def testModifyConfigValue(self):
        if False:
            for i in range(10):
                print('nop')
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        config.set('graph_recursion_depth', 9)
        config.set('mouse_mode', False)
        self.assertEqual(9, config.get('graph_recursion_depth'))
        self.assertEqual(False, config.get('mouse_mode'))

    def testModifyConfigValueWithTypeCasting(self):
        if False:
            print('Hello World!')
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        config.set('graph_recursion_depth', '18')
        config.set('mouse_mode', 'false')
        self.assertEqual(18, config.get('graph_recursion_depth'))
        self.assertEqual(False, config.get('mouse_mode'))

    def testModifyConfigValueWithTypeCastingFailure(self):
        if False:
            return 10
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        with self.assertRaises(ValueError):
            config.set('mouse_mode', 'maybe')

    def testLoadFromModifiedConfigFile(self):
        if False:
            return 10
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        config.set('graph_recursion_depth', 9)
        config.set('mouse_mode', False)
        config2 = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        self.assertEqual(9, config2.get('graph_recursion_depth'))
        self.assertEqual(False, config2.get('mouse_mode'))

    def testSummarizeFromConfig(self):
        if False:
            for i in range(10):
                print('nop')
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        output = config.summarize()
        self.assertEqual(['Command-line configuration:', '', '  graph_recursion_depth: %d' % config.get('graph_recursion_depth'), '  mouse_mode: %s' % config.get('mouse_mode')], output.lines)

    def testSummarizeFromConfigWithHighlight(self):
        if False:
            i = 10
            return i + 15
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        output = config.summarize(highlight='mouse_mode')
        self.assertEqual(['Command-line configuration:', '', '  graph_recursion_depth: %d' % config.get('graph_recursion_depth'), '  mouse_mode: %s' % config.get('mouse_mode')], output.lines)
        self.assertEqual((2, 12, ['underline', 'bold']), output.font_attr_segs[3][0])
        self.assertEqual((14, 18, 'bold'), output.font_attr_segs[3][1])

    def testSetCallback(self):
        if False:
            while True:
                i = 10
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        test_value = {'graph_recursion_depth': -1}

        def callback(config):
            if False:
                print('Hello World!')
            test_value['graph_recursion_depth'] = config.get('graph_recursion_depth')
        config.set_callback('graph_recursion_depth', callback)
        config.set('graph_recursion_depth', config.get('graph_recursion_depth') - 1)
        self.assertEqual(test_value['graph_recursion_depth'], config.get('graph_recursion_depth'))

    def testSetCallbackInvalidPropertyName(self):
        if False:
            print('Hello World!')
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        with self.assertRaises(KeyError):
            config.set_callback('nonexistent_property_name', print)

    def testSetCallbackNotCallable(self):
        if False:
            return 10
        config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
        with self.assertRaises(TypeError):
            config.set_callback('graph_recursion_depth', 1)
if __name__ == '__main__':
    googletest.main()