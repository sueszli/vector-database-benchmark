import unittest
import paddle

class TestDispensable(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.base.core._set_prim_all_enabled(True)

    def tearDown(self):
        if False:
            while True:
                i = 10
        paddle.base.core._set_prim_all_enabled(False)

    def test_dispensable(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                print('Hello World!')
            return paddle.split(x, num_or_sections=2)
        f = paddle.jit.to_static(full_graph=True)(f)
        x = paddle.rand((8,))
        x.stop_gradient = False
        op = f.get_concrete_program(x)[1].backward_program.block(0).ops[-1]
        self.assertEqual(op.attr('op_role'), int(paddle.base.core.op_proto_and_checker_maker.OpRole.Backward))
        self.assertIn('AxisTensor', op.input_names)
if __name__ == '__main__':
    unittest.main()