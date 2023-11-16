"""Test cases for debug XLA dumps."""
import glob
import os
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class XlaDumpToSpongeTest(xla_test.XLATestCase):
    """Test that ensures --XLA_FLAGS=--dump_to_xla=sponge produces output."""

    def _compute(self):
        if False:
            while True:
                i = 10
        with self.session() as sess, self.device_scope():
            data = np.array([0], dtype=np.float32)
            indices = np.array([0], dtype=np.int32)
            d = array_ops.placeholder(data.dtype, shape=data.shape)
            i = array_ops.placeholder(indices.dtype, shape=indices.shape)
            sess.run(math_ops.segment_max_v2(data, indices, 1), {d: data, i: indices})

    def testDumpToSponge(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['XLA_FLAGS'] = '--xla_dump_to=sponge'
        self._compute()
        out_dir = os.environ['TEST_UNDECLARED_OUTPUTS_DIR']
        self.assertNotEmpty(glob.glob(os.path.join(out_dir, 'module_0*')))
if __name__ == '__main__':
    googletest.main()