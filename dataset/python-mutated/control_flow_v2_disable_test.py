"""Tests that TF2_BEHAVIOR=1 and TF_ENABLE_CONTROL_FLOW_V2=0 disables cfv2."""
import os
os.environ['TF2_BEHAVIOR'] = '1'
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '0'
from tensorflow.python import tf2
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

class ControlFlowV2DisableTest(test.TestCase):

    def testIsDisabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(tf2.enabled())
        self.assertFalse(control_flow_util.ENABLE_CONTROL_FLOW_V2)
if __name__ == '__main__':
    googletest.main()