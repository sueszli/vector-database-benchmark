import unittest
import paddle
from paddle.base import core, framework

class TestGetGradOpDescPrimEnabled(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fwd_type = 'tanh'
        self.inputs = {'X': ['x']}
        self.outputs = {'Out': ['y']}
        self.no_grad_var = set()
        self.grad_sub_block = ()
        self.desired_ops = 'tanh_grad'
        self.desired_ops_no_skip = ('elementwise_mul', 'fill_constant', 'elementwise_sub', 'elementwise_mul')
        paddle.enable_static()
        block = framework.Block(framework.Program(), 0)
        block.append_op(type=self.fwd_type, inputs={n: [block.create_var(name=v, stop_gradient=False) for v in vs] for (n, vs) in self.inputs.items()}, outputs={n: [block.create_var(name=v, stop_gradient=False) for v in vs] for (n, vs) in self.outputs.items()})
        for (_, outs) in self.outputs.items():
            for out in outs:
                block.create_var(name=out + core.grad_var_suffix())
        self.fwd = block.ops[0].desc

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()

    def test_get_grad_op_desc_without_skip(self):
        if False:
            return 10
        core._set_prim_backward_enabled(True)
        actual = tuple((desc.type() for desc in core.get_grad_op_desc(self.fwd, self.no_grad_var, self.grad_sub_block)[0]))
        self.assertEqual(actual, self.desired_ops_no_skip)
        core._set_prim_backward_enabled(False)

    def test_get_grad_op_desc_with_skip(self):
        if False:
            print('Hello World!')
        core._set_prim_backward_enabled(True)
        core._add_skip_comp_ops('tanh')
        actual = tuple((desc.type() for desc in core.get_grad_op_desc(self.fwd, self.no_grad_var, self.grad_sub_block)[0]))
        core._remove_skip_comp_ops('tanh')
        self.assertEqual(actual[0], self.desired_ops)
        core._set_prim_backward_enabled(False)
if __name__ == '__main__':
    unittest.main()