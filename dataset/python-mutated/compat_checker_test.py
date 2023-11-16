"""Tests for version compatibility checker for TensorFlow Builder."""
import os
import unittest
from tensorflow.tools.tensorflow_builder.compat_checker import compat_checker
PATH_TO_DIR = 'tensorflow/tools/tensorflow_builder/compat_checker'
USER_CONFIG_IN_RANGE = {'apple': ['1.0'], 'banana': ['3'], 'kiwi': ['2.0'], 'watermelon': ['2.0.0'], 'orange': ['4.1'], 'cherry': ['1.5'], 'cranberry': ['1.0'], 'raspberry': ['3.0'], 'tangerine': ['2.0.0'], 'jackfruit': ['1.0'], 'grapefruit': ['2.0'], 'apricot': ['wind', 'flower'], 'grape': ['7.1'], 'blueberry': ['3.0']}
USER_CONFIG_NOT_IN_RANGE = {'apple': ['4.0'], 'banana': ['5'], 'kiwi': ['3.5'], 'watermelon': ['5.0'], 'orange': ['3.5'], 'cherry': ['2.0'], 'raspberry': ['-1'], 'cranberry': ['4.5'], 'tangerine': ['0'], 'jackfruit': ['5.0'], 'grapefruit': ['2.5'], 'apricot': ['hello', 'world'], 'blueberry': ['11.0'], 'grape': ['7.0'], 'cantaloupe': ['11.0']}
USER_CONFIG_MISSING = {'avocado': ['3.0'], 'apple': [], 'banana': ''}

class CompatCheckerTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        'Set up test.'
        super(CompatCheckerTest, self).setUp()
        self.test_file = os.path.join(PATH_TO_DIR, 'test_config.ini')

    def testWithUserConfigInRange(self):
        if False:
            return 10
        'Test a set of configs that are supported.\n\n    Testing with the following combination should always return `success`:\n      [1] A set of configurations that are supported and/or compatible.\n      [2] `.ini` config file with proper formatting.\n    '
        self.compat_checker = compat_checker.ConfigCompatChecker(USER_CONFIG_IN_RANGE, self.test_file)
        self.assertTrue(self.compat_checker.check_compatibility())
        self.assertFalse(len(self.compat_checker.error_msg))
        cnt = len(list(USER_CONFIG_IN_RANGE.keys()))
        self.assertEqual(len(self.compat_checker.successes), cnt)

    def testWithUserConfigNotInRange(self):
        if False:
            while True:
                i = 10
        'Test a set of configs that are NOT supported.\n\n    Testing with the following combination should always return `failure`:\n      [1] A set of configurations that are NOT supported and/or compatible.\n      [2] `.ini` config file with proper formatting.\n    '
        self.compat_checker = compat_checker.ConfigCompatChecker(USER_CONFIG_NOT_IN_RANGE, self.test_file)
        self.assertFalse(self.compat_checker.check_compatibility())
        err_msg_list = self.compat_checker.failures
        self.assertTrue(len(err_msg_list))
        cnt = len(list(USER_CONFIG_NOT_IN_RANGE.keys()))
        self.assertEqual(len(err_msg_list), cnt)

    def testWithUserConfigMissing(self):
        if False:
            print('Hello World!')
        'Test a set of configs that are empty or missing specification.'
        self.compat_checker = compat_checker.ConfigCompatChecker(USER_CONFIG_MISSING, self.test_file)
        self.assertFalse(self.compat_checker.check_compatibility())
if __name__ == '__main__':
    unittest.main()