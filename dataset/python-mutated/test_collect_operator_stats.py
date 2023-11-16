import unittest
from amp_base_models import build_while_model
import paddle

class TestOpStatsEager(unittest.TestCase):

    def _check_result(self, dtype):
        if False:
            i = 10
            return i + 15
        op_list = paddle.base.core.get_low_precision_op_list()
        self.assertTrue('elementwise_add' in op_list)
        self.assertTrue('conv2d' in op_list)
        conv2d_called = op_list['conv2d'].split(',')
        add_called = op_list['elementwise_add'].split(',')
        add_num = 0
        conv_num = 0
        for i in range(4):
            add_num += int(add_called[i])
            conv_num += int(add_called[i])
        self.assertTrue(conv_num == 1)
        self.assertTrue(add_num == 1)
        if dtype == 'float16':
            self.assertTrue(int(conv2d_called[0]) == 1)
            self.assertTrue(int(add_called[0]) == 1)

    def test_enable_disable(self):
        if False:
            i = 10
            return i + 15
        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])
        paddle.amp.debugging.enable_operator_stats_collection()
        with paddle.amp.auto_cast(enable=True, level='O2'):
            out = conv(x)
        paddle.amp.debugging.disable_operator_stats_collection()
        self._check_result(dtype=out.dtype)

    def test_context(self):
        if False:
            for i in range(10):
                print('nop')
        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])
        with paddle.amp.debugging.collect_operator_stats():
            with paddle.amp.auto_cast(enable=True, level='O2'):
                out = conv(x)
        self._check_result(dtype=out.dtype)

class TestOpStatsStatic(unittest.TestCase):

    def test_while_op(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        (main_program, startup_program) = build_while_model()
        self.assertEqual(main_program.num_blocks, 2)
        paddle.static.amp.debugging.collect_operator_stats(program=main_program, print_subblocks=True)
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()