import unittest
import numpy as np
import paddle

class TestCheckFetchList(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.feed = {'x': np.array([[0], [0], [1], [0]], dtype='float32')}
        self.expected = np.array([[0], [1], [0]], dtype='float32')
        self.build_program()
        self.exe = paddle.static.Executor(paddle.CPUPlace())

    def build_program(self):
        if False:
            return 10
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=[4, 1], dtype='float32')
            output = paddle.unique_consecutive(x, return_inverse=True, return_counts=True, axis=0)
        self.main_program = main_program
        self.fetch_list = output

    def test_with_tuple(self):
        if False:
            return 10
        res = self.exe.run(self.main_program, feed=self.feed, fetch_list=[self.fetch_list], return_numpy=True)
        np.testing.assert_array_equal(res[0], self.expected)

    def test_with_error(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            fetch_list = [23]
            res = self.exe.run(self.main_program, feed=self.feed, fetch_list=fetch_list)
        with self.assertRaises(TypeError):
            fetch_list = [(self.fetch_list[0], 32)]
            res = self.exe.run(self.main_program, feed=self.feed, fetch_list=fetch_list)
if __name__ == '__main__':
    unittest.main()