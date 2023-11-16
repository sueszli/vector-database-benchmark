import unittest
import paddle
from paddle import base

class TestNameScope(unittest.TestCase):

    def test_name_scope(self):
        if False:
            return 10
        with base.name_scope('s1'):
            a = paddle.static.data(name='data', shape=[-1, 1], dtype='int32')
            b = a + 1
            with base.name_scope('s2'):
                c = b * 1
            with base.name_scope('s3'):
                d = c / 1
        with base.name_scope('s1'):
            f = paddle.pow(d, 2.0)
        with base.name_scope('s4'):
            g = f - 1
        for op in base.default_main_program().block(0).ops:
            if op.type == 'elementwise_add':
                self.assertEqual(op.desc.attr('op_namescope'), '/s1/')
            elif op.type == 'elementwise_mul':
                self.assertEqual(op.desc.attr('op_namescope'), '/s1/s2/')
            elif op.type == 'elementwise_div':
                self.assertEqual(op.desc.attr('op_namescope'), '/s1/s3/')
            elif op.type == 'elementwise_sub':
                self.assertEqual(op.desc.attr('op_namescope'), '/s4/')
            elif op.type == 'pow':
                self.assertEqual(op.desc.attr('op_namescope'), '/s1_1/')
if __name__ == '__main__':
    unittest.main()