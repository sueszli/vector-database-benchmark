import unittest
import paddle
from paddle import base

class TestAttrSet(unittest.TestCase):

    def test_set_bool_attr(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name='x', shape=[-1, 3, 7, 3, 7], dtype='float32')
        param_attr = base.ParamAttr(name='batch_norm_w', initializer=paddle.nn.initializer.Constant(value=1.0))
        bias_attr = base.ParamAttr(name='batch_norm_b', initializer=paddle.nn.initializer.Constant(value=0.0))
        bn = paddle.static.nn.batch_norm(input=x, param_attr=param_attr, bias_attr=bias_attr)
        block = base.default_main_program().desc.block(0)
        op = block.op(0)
        before_type = op.attr_type('is_test')
        op._set_attr('is_test', True)
        after_type = op.attr_type('is_test')
        self.assertEqual(before_type, after_type)
if __name__ == '__main__':
    unittest.main()