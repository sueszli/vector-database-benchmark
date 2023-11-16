"""Tests for forward and backwards compatibility utilities."""
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python.platform import test

class DisableV2BehaviorTest(test.TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        t = constant_op.constant([1, 2, 3])
        self.assertTrue(isinstance(t, ops.EagerTensor))
        t = _pywrap_tf2.is_enabled()
        self.assertTrue(t)
        v2_compat.disable_v2_behavior()
        t = constant_op.constant([1, 2, 3])
        self.assertFalse(isinstance(t, ops.EagerTensor))
        t = _pywrap_tf2.is_enabled()
        self.assertFalse(t)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()