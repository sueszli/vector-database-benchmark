import unittest
from paddle.base import core
core._set_prim_backward_enabled(True)
import parameterized as param
import paddle
from paddle.base import core, framework

@param.parameterized_class(('fwd_type', 'inputs', 'outputs', 'no_grad_var', 'grad_sub_block', 'desired_ops'), (('tanh', {'X': ['x']}, {'Out': ['y']}, set(), (), ('elementwise_mul', 'fill_constant', 'elementwise_sub', 'elementwise_mul')), ('empty', {}, {'Out': ['y']}, set(), (), ())))
class TestGetGradOpDescPrimEnabled(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        paddle.enable_static()
        block = framework.Block(framework.Program(), 0)
        block.append_op(type=cls.fwd_type, inputs={n: [block.create_var(name=v, stop_gradient=False) for v in vs] for (n, vs) in cls.inputs.items()}, outputs={n: [block.create_var(name=v, stop_gradient=False) for v in vs] for (n, vs) in cls.outputs.items()})
        for (_, outs) in cls.outputs.items():
            for out in outs:
                block.create_var(name=out + core.grad_var_suffix())
        cls.fwd = block.ops[0].desc

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        paddle.disable_static()

    def test_get_grad_op_desc(self):
        if False:
            while True:
                i = 10
        actual = tuple((desc.type() for desc in core.get_grad_op_desc(self.fwd, self.no_grad_var, self.grad_sub_block)[0]))
        self.assertEqual(actual, self.desired_ops)
        core._set_prim_backward_enabled(False)
if __name__ == '__main__':
    unittest.main()