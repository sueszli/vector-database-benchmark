import unittest
import paddle
from paddle import pir
from paddle.decomposition import decompose
from paddle.framework import core
paddle.enable_static()

def get_ir_program():
    if False:
        for i in range(10):
            print('nop')
    paddle.enable_static()
    x = paddle.randn([4, 4])
    (main_program, start_program) = (paddle.static.Program(), paddle.static.Program())
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        y_s = paddle.matmul(x_s, x_s)
        y_s = paddle.add(x_s, y_s)
        y_s = paddle.mean(y_s)
        y_s = paddle.tanh(y_s)
    pir_program = pir.translate_to_pir(main_program.desc)
    return pir_program

class TestBuildOp(unittest.TestCase):

    def test_build_op(self):
        if False:
            return 10
        pir_program = get_ir_program()
        y = pir_program.global_block().ops[-2].results()
        orig_shape = y[0].shape
        with paddle.pir_utils.IrGuard():
            core._set_prim_forward_enabled(True)
            y_new = decompose(pir_program, y)
            core._set_prim_forward_enabled(False)
            new_shape = y_new[0].shape
            assert orig_shape == new_shape, f'Original shape {orig_shape} is not equal to new shape {new_shape}'
            op_name_list = [op.name() for op in pir_program.global_block().ops]
            self.assertEqual(op_name_list, ['pd_op.data', 'pd_op.matmul', 'pd_op.add', 'pd_op.full_int_array', 'pd_op.sum', 'pd_op.full', 'pd_op.divide', 'pd_op.tanh'])
if __name__ == '__main__':
    unittest.main()