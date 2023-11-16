import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class TraceTest(test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        x = np.random.seed(0)

    def compare(self, x):
        if False:
            return 10
        np_ans = np.trace(x, axis1=-2, axis2=-1)
        with self.cached_session():
            tf_ans = math_ops.trace(x).eval()
        self.assertAllClose(tf_ans, np_ans)

    @test_util.run_deprecated_v1
    def testTrace(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [np.int32, np.float32, np.float64]:
            for shape in [[2, 2], [2, 3], [3, 2], [2, 3, 2], [2, 2, 2, 3]]:
                x = np.random.rand(np.prod(shape)).astype(dtype).reshape(shape)
                self.compare(x)
if __name__ == '__main__':
    test.main()