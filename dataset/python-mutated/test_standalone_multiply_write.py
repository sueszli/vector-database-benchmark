import unittest
from test_standalone_controlflow import TestCompatibility
import paddle
from paddle.base.framework import Program
paddle.enable_static()

class TestMultiplyWrite(TestCompatibility):

    def _get_feed(self):
        if False:
            i = 10
            return i + 15
        'return the feeds'
        return None

    def build_program(self):
        if False:
            return 10
        main_program = Program()
        startup_program = Program()
        with paddle.static.program_guard(main_program, startup_program):
            out = paddle.full((1,), 1)
            inp1 = paddle.full((1,), 2)
            inp2 = paddle.full((1,), 3)
            paddle.assign(inp1, out)
            paddle.assign(inp2, out)
        return (main_program, startup_program, out)

    def run_dygraph_once(self, feed):
        if False:
            return 10
        out = paddle.full((1,), 1)
        inp1 = paddle.full((1,), 2)
        inp2 = paddle.full((1,), 3)
        paddle.assign(inp1, out)
        paddle.assign(inp2, out)
        return [out.numpy()]

    def setUp(self):
        if False:
            print('Hello World!')
        self.place = paddle.CPUPlace()
        self.iter_run = 5
if __name__ == '__main__':
    unittest.main()