"""Tests lowering of tf.bitcast"""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test

class CastOpsTest(xla_test.XLATestCase):

    def testBitcastToLarger(self):
        if False:
            while True:
                i = 10
        with ops.device('device:{}:0'.format(self.device)):

            def f(x):
                if False:
                    i = 10
                    return i + 15
                t = array_ops.bitcast(x, dtypes.float32)
                return math_ops.reduce_sum(t, axis=1)
            compiled_f = def_function.function(f, jit_compile=True)
            x = random_ops.random_normal([10, 10, 2], dtype=dtypes.float16)
            with ops.device(self.device):
                out = f(x)
                compiled_out = compiled_f(x)
                self.assertAllClose(out, compiled_out)
                self.assertEqual(out.shape[0], 10)
            hlo = compiled_f.experimental_get_compiler_ir(x)(stage='hlo')
            self.assertIn('f32[10,10]{1,0} bitcast-convert(f16[10,10,2]{2,1,0}', hlo)

    def testBitcastToSmaller(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()