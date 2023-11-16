import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.incubate.autograd import primapi
paddle.framework.random._manual_program_seed(2023)

def fn(x):
    if False:
        return 10
    dropout1 = paddle.nn.Dropout(p=0.5)
    dropout2 = paddle.nn.Dropout(p=0.6)
    y = dropout1(x)
    z = dropout2(y)
    return z

class TestCompositeCopyOp(unittest.TestCase):
    """This case is set to test copying op process even if some attrs of origin op has been blocked during constructing program."""

    def cal_composite(self, inputs):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data('x', shape=inputs.shape, dtype=str(inputs.dtype))
            y = fn(x)
            blocks = main_program.blocks
            fwd_ops = [op.type for op in blocks[0].ops]
            self.assertTrue('dropout' in fwd_ops)
            primapi.to_prim(blocks)
            fwd_ops_new = [op.type for op in blocks[0].ops]
            self.assertTrue('dropout' in fwd_ops_new)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def test_forward(self):
        if False:
            print('Hello World!')
        core._set_prim_forward_blacklist('dropout')
        np_data = np.random.random([16, 64, 128, 128]).astype('float32')
        tensor_data = paddle.to_tensor(np_data)
        expect = fn(tensor_data).numpy()
        actual = self.cal_composite(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=0, atol=0)
if __name__ == '__main__':
    unittest.main()