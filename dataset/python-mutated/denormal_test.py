"""Tests for denormal handling."""
import os
import platform
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

@test_util.with_eager_op_as_function
class DenormalTest(test.TestCase):

    def testPythonHasDenormals(self):
        if False:
            i = 10
            return i + 15
        'Non-tf numpy code should treat denormals correctly.'
        for dtype in (np.float32, np.float64):
            tiny = np.finfo(dtype).tiny
            self.assertEqual(tiny, tiny / 16 * 16)

    def _flushDenormalsTest(self, dtypes):
        if False:
            for i in range(10):
                print('nop')
        if platform.machine() == 'ppc64le' or platform.machine() == 's390x' or platform.machine() == 'aarch64':
            return
        for dtype in dtypes:
            tiny = np.finfo(dtype).tiny
            for shape in ((), (1 << 20,)):
                flush = 0.1 * constant_op.constant(tiny, shape=shape)
                self.assertAllEqual(self.evaluate(flush), np.zeros(shape))
                self.testPythonHasDenormals()

    @test_util.run_in_graph_and_eager_modes(use_gpu=False)
    def testFlushDenormalsCPU(self):
        if False:
            while True:
                i = 10
        self._flushDenormalsTest(dtypes=(np.float32, np.float64))

    @test_util.run_in_graph_and_eager_modes(use_gpu=True)
    def testFlushDenormalsGPU(self):
        if False:
            return 10
        self._flushDenormalsTest(dtypes=(np.float32,))
if __name__ == '__main__':
    original_xla_flags = os.environ.get('XLA_FLAGS')
    new_xla_flags = '--xla_gpu_ftz=true'
    if original_xla_flags:
        new_xla_flags = new_xla_flags + ' ' + original_xla_flags
    os.environ['XLA_FLAGS'] = new_xla_flags
    test.main()