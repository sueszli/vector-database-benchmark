"""Tests for tensorflow.compiler.mlir.tfr.examples.customization.ops_defs.py."""
import os
import tensorflow as tf
from tensorflow.compiler.mlir.tfr.python import test_utils
from tensorflow.python.framework import test_ops
from tensorflow.python.platform import test

class TestOpsDefsTest(test_utils.OpsDefsTest):

    def test_test_ops(self):
        if False:
            for i in range(10):
                print('nop')
        attr = tf.function(test_ops.test_attr)(tf.float32)
        self.assertAllClose(attr.numpy(), 100.0)
if __name__ == '__main__':
    os.environ['TF_MLIR_TFR_LIB_DIR'] = 'tensorflow/compiler/mlir/tfr/examples/customization'
    test.main()