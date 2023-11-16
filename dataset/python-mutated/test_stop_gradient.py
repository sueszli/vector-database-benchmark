import unittest
import paddle

class TestAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.enable_static()

    def assert_api(self, api_func, stop_gradient):
        if False:
            print('Hello World!')
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = api_func()
        self.assertEqual(x.stop_gradient, stop_gradient)
        x.stop_gradient = not stop_gradient
        self.assertEqual(x.stop_gradient, not stop_gradient)

    def test_full(self):
        if False:
            print('Hello World!')
        api = lambda : paddle.full(shape=[2, 3], fill_value=1.0)
        self.assert_api(api, True)

    def test_data(self):
        if False:
            for i in range(10):
                print('nop')
        api = lambda : paddle.static.data('x', [4, 4], dtype='float32')
        self.assert_api(api, True)

class TestParametes(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()

    def test_create_param(self):
        if False:
            print('Hello World!')
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            w = paddle.create_parameter(shape=[784, 200], dtype='float32')
        self.assertEqual(w.stop_gradient, False)
        self.assertEqual(w.persistable, True)
        w.stop_gradient = True
        w.persistable = False
        self.assertEqual(w.stop_gradient, True)
        self.assertEqual(w.persistable, False)
if __name__ == '__main__':
    unittest.main()