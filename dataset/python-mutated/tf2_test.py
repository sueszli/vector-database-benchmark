"""Tests for enabling and disabling TF2 behavior."""
from absl.testing import parameterized
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python.platform import test

class EnablingTF2Behavior(test.TestCase, parameterized.TestCase):

    def __init__(self, methodName):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(methodName)
        self._set_default_seed = False

    @combinations.generate(test_base.v1_only_combinations())
    def test_tf1_enable_tf2_behaviour(self):
        if False:
            while True:
                i = 10
        self.assertFalse(tf2.enabled())
        self.assertFalse(_pywrap_tf2.is_enabled())
        v2_compat.enable_v2_behavior()
        self.assertTrue(tf2.enabled())
        self.assertTrue(_pywrap_tf2.is_enabled())
        v2_compat.disable_v2_behavior()
        self.assertFalse(tf2.enabled())
        self.assertFalse(_pywrap_tf2.is_enabled())

    @combinations.generate(test_base.v1_only_combinations())
    def test_tf1_disable_tf2_behaviour(self):
        if False:
            while True:
                i = 10
        self.assertFalse(tf2.enabled())
        self.assertFalse(_pywrap_tf2.is_enabled())
        v2_compat.disable_v2_behavior()
        self.assertFalse(tf2.enabled())
        self.assertFalse(_pywrap_tf2.is_enabled())
        v2_compat.enable_v2_behavior()
        self.assertTrue(tf2.enabled())
        self.assertTrue(_pywrap_tf2.is_enabled())

    @combinations.generate(test_base.v2_only_combinations())
    def test_tf2_enable_tf2_behaviour(self):
        if False:
            print('Hello World!')
        self.assertTrue(tf2.enabled())
        self.assertTrue(_pywrap_tf2.is_enabled())
        v2_compat.enable_v2_behavior()
        self.assertTrue(tf2.enabled())
        self.assertTrue(_pywrap_tf2.is_enabled())
        v2_compat.disable_v2_behavior()
        self.assertFalse(tf2.enabled())
        self.assertFalse(_pywrap_tf2.is_enabled())

    @combinations.generate(test_base.v2_only_combinations())
    def test_tf2_disable_tf2_behaviour(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(tf2.enabled())
        self.assertTrue(_pywrap_tf2.is_enabled())
        v2_compat.disable_v2_behavior()
        self.assertFalse(tf2.enabled())
        self.assertFalse(_pywrap_tf2.is_enabled())
        v2_compat.enable_v2_behavior()
        self.assertTrue(tf2.enabled())
        self.assertTrue(_pywrap_tf2.is_enabled())
if __name__ == '__main__':
    test.main()