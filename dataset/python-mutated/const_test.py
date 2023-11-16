"""Tests for const op compilation."""
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class ConstOpTest(test_util.TensorFlowTestCase):

    def testConst(self):
        if False:
            print('Hello World!')
        types = {dtypes.bool, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64, dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64, dtypes.float8_e5m2, dtypes.float8_e4m3fn}
        for dtype in types:
            with self.subTest(dtype=dtype):
                if dtype == dtypes.bool:
                    values = [True, False]
                else:
                    values = [0.0, 1.0, -1.0, dtype.min, dtype.max]
                if dtype.is_floating:
                    values.extend([float('Inf'), -float('Inf'), float('NaN')])
                values = np.array(values, dtype=dtype.as_numpy_dtype)

                @def_function.function(jit_compile=True)
                def f():
                    if False:
                        for i in range(10):
                            print('nop')
                    return constant_op.constant(values, dtype)
                result = f()
                self.assertAllEqual(self.evaluate(result), values)
if __name__ == '__main__':
    test.main()