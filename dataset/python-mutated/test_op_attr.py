import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only
import paddle
from paddle.static import InputSpec

class MySub(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x, y, name=None):
        if False:
            for i in range(10):
                print('nop')
        return paddle.subtract(x, y, name)

class NetWithOpAttr(paddle.nn.Layer):

    def __init__(self, in_num, out_num):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear = paddle.nn.Linear(in_num, out_num)
        self.bn = paddle.nn.BatchNorm(out_num)
        self.sub = MySub()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        out = self.linear(x)
        out = self.sub(out, x)
        out = self.bn(out)
        return out

    @paddle.jit.to_static(input_spec=[InputSpec([10, 16])], full_graph=True)
    def with_cond(self, x):
        if False:
            print('Hello World!')
        if paddle.mean(x) > 0.0:
            out = self.linear(x)
        else:
            out = self.sub(x, x)
        out = self.bn(out)
        return out

class CheckOpAttr(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        self.in_num = 16
        self.out_num = 16
        self.x = paddle.randn([10, self.in_num])
        self.expected_results()

    def expected_results(self):
        if False:
            print('Hello World!')
        self.fc_attrs = {'int_val': 10, 'int_vals': [10, 20], 'float_val': 3.8, 'float_vals': [3.8, -0.2]}
        self.bn_attrs = {'bool_val': True, 'bool_vals': [True, False]}
        self.sub_attrs = {'int_vals': [10, 20], 'bool_vals': [True, False]}
        self.infos = {'matmul': self.fc_attrs, 'elementwise_add': self.fc_attrs, 'batch_norm': self.bn_attrs, 'tanh': self.bn_attrs, 'elementwise_sub': self.sub_attrs}

    @test_ast_only
    def test_set_op_attrs(self):
        if False:
            return 10
        net = NetWithOpAttr(self.in_num, self.out_num)
        net.linear._set_op_attrs(self.fc_attrs)
        net.bn._set_op_attrs({'bool_val': False})
        net.bn._set_op_attrs(self.bn_attrs)
        net.sub._set_op_attrs(self.sub_attrs)
        self.assertEqual(len(net.linear._forward_pre_hooks), 1)
        self.assertEqual(len(net.linear._forward_post_hooks), 1)
        net = paddle.jit.to_static(net, input_spec=[InputSpec.from_tensor(self.x)])
        self.check_op_attrs(net.forward.concrete_program.main_program)
        self.assertEqual(len(net.linear._forward_pre_hooks), 0)
        self.assertEqual(len(net.linear._forward_post_hooks), 0)

    def check_op_attrs(self, main_program):
        if False:
            while True:
                i = 10
        for cur_block in main_program.blocks:
            ops = cur_block.ops
            for op in ops:
                if op.type not in self.infos:
                    continue
                for (attr_name, expect_vals) in self.infos[op.type].items():
                    op_vals = op.desc.attr(attr_name)
                    if not isinstance(expect_vals, list):
                        expect_vals = [expect_vals]
                        op_vals = [op_vals]
                    for (op_val, expect_val) in zip(op_vals, expect_vals):
                        if isinstance(op_val, float):
                            self.assertAlmostEqual(op_val, expect_val)
                        else:
                            self.assertEqual(op_val, expect_val)

    @test_ast_only
    def test_set_op_attrs_with_sub_block(self):
        if False:
            for i in range(10):
                print('nop')
        net = NetWithOpAttr(self.in_num, self.out_num)
        net.linear._set_op_attrs({'int_vals': [0, 0]})
        net.linear._set_op_attrs(self.fc_attrs)
        net.bn._set_op_attrs(self.bn_attrs)
        net.sub._set_op_attrs(self.sub_attrs)
        self.assertEqual(len(net.linear._forward_pre_hooks), 1)
        self.assertEqual(len(net.linear._forward_post_hooks), 1)
        self.check_op_attrs(net.with_cond.concrete_program.main_program)
        self.assertEqual(len(net.linear._forward_pre_hooks), 0)
        self.assertEqual(len(net.linear._forward_post_hooks), 0)

    def test_type_error(self):
        if False:
            while True:
                i = 10
        net = NetWithOpAttr(self.in_num, self.out_num)
        with self.assertRaises(TypeError):
            net.linear._set_op_attrs([self.fc_attrs])
if __name__ == '__main__':
    unittest.main()