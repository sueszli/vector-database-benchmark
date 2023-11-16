"""Tests for compilation that involves constant arguments."""
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

class ConstArgTest(xla_test.XLATestCase):

    def testValueInference(self):
        if False:
            print('Hello World!')
        with self.session() as session:
            with self.test_scope():
                a = array_ops.placeholder(dtypes.int32, [], name='a')
                size = array_ops.reshape(array_ops.where_v2(a >= 0, 1, 0), [1])
                output = xla.dynamic_slice([11, 12, 13], [0], size)
            result = session.run(output, {a: 1})
            expected = [11]
            self.assertEqual(result, expected)
if __name__ == '__main__':
    googletest.main()