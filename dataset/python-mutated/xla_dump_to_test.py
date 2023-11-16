"""Test cases for debug XLA dumps."""
import glob
import os
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class XlaDumpToDirTest(xla_test.XLATestCase):
    """Test that ensures --XLA_FLAGS=--dump_to_xla=<dir> produces output."""

    def _compute(self):
        if False:
            return 10
        with self.session() as sess, self.device_scope():
            data = np.array([0], dtype=np.float32)
            indices = np.array([0], dtype=np.int32)
            d = array_ops.placeholder(data.dtype, shape=data.shape)
            i = array_ops.placeholder(indices.dtype, shape=indices.shape)
            sess.run(math_ops.segment_max_v2(data, indices, 1), {d: data, i: indices})

    def testDumpToTempDir(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.create_tempdir().full_path
        os.environ['XLA_FLAGS'] = '--xla_dump_to=' + tmp_dir
        self._compute()
        self.assertNotEmpty(glob.glob(os.path.join(tmp_dir, 'module_0*')))
if __name__ == '__main__':
    googletest.main()