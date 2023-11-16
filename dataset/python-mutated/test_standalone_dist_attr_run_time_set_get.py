import sys
import unittest
import paddle
from paddle.static import Program, program_guard
paddle.enable_static()

class TestOpProfiling(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def _build_startup_program_and_train_program(self):
        if False:
            for i in range(10):
                print('nop')
        startup_program = Program()
        train_program = Program()
        with program_guard(train_program, startup_program):
            data = paddle.static.data(name='X', shape=[1024, 1], dtype='float32')
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
        return (startup_program, train_program, loss)

    def test_run_time_us_set_get_method(self):
        if False:
            print('Hello World!')
        '\n        * test if the newly added "run_time_us_" actually works (set then get)\n        '
        (startup_program, train_program, loss) = self._build_startup_program_and_train_program()
        global_block = startup_program.global_block()
        global_block.ops[0].dist_attr.run_time_us = 1.0
        dt = global_block.ops[0].dist_attr.run_time_us
        if dt != 1.0:
            raise RuntimeError('dist_attr set/get method failed!')
        else:
            sys.stdout.write('OK.')
if __name__ == '__main__':
    unittest.main()