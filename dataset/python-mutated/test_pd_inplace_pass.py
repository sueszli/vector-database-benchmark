import unittest
import numpy as np
import paddle
paddle.enable_static()

class TestPdInplacePass(unittest.TestCase):

    def test_pd_inplace_pass(self):
        if False:
            i = 10
            return i + 15
        place = paddle.framework.core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.static.data('x', [2, 2], dtype='float32')
                y = paddle.ones([2, 2], dtype='float32')
                z = paddle.divide(x, y)
                out = paddle.nn.functional.relu(z)
                exe = paddle.static.Executor()
                x_feed = np.ones([2, 2], dtype=np.float32) * 10
                (sum_value,) = exe.run(feed={'x': x_feed}, fetch_list=[out])
                self.assertEqual((sum_value == np.ones([2, 2], dtype='float32') * 10).all(), True)
if __name__ == '__main__':
    unittest.main()