"""Tests that TF2_BEHAVIOR=1 enables cfv2."""
import os
os.environ['TF2_BEHAVIOR'] = '1'
from tensorflow.python import tf2
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

class ControlFlowV2EnableTest(test.TestCase):

    def testIsEnabled(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(tf2.enabled())
        self.assertTrue(control_flow_util.ENABLE_CONTROL_FLOW_V2)
if __name__ == '__main__':
    googletest.main()