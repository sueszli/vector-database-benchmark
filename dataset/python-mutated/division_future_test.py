"""Tests for division with division imported from __future__.

This file should be exactly the same as division_past_test.py except
for the __future__ division line.
"""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test

class DivisionTestCase(test.TestCase):

    def testDivision(self):
        if False:
            i = 10
            return i + 15
        'Test all the different ways to divide.'
        values = [1, 2, 7, 11]
        functions = (lambda x: x, constant_op.constant)
        dtypes = (np.int32, np.int64, np.float32, np.float64)
        tensors = []
        checks = []

        def check(x, y):
            if False:
                print('Hello World!')
            x = ops.convert_to_tensor(x)
            y = ops.convert_to_tensor(y)
            tensors.append((x, y))

            def f(x, y):
                if False:
                    while True:
                        i = 10
                self.assertEqual(x.dtype, y.dtype)
                self.assertAllClose(x, y)
            checks.append(f)
        with self.cached_session() as sess:
            for dtype in dtypes:
                for x in map(dtype, values):
                    for y in map(dtype, values):
                        for fx in functions:
                            for fy in functions:
                                tf_x = fx(x)
                                tf_y = fy(y)
                                div = x / y
                                tf_div = tf_x / tf_y
                                check(div, tf_div)
                                floordiv = x // y
                                tf_floordiv = tf_x // tf_y
                                check(floordiv, tf_floordiv)
            for (f, (x, y)) in zip(checks, self.evaluate(tensors)):
                f(x, y)
if __name__ == '__main__':
    test.main()