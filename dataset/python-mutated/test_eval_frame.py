import collections
import unittest
from sys import version_info
import paddle

class TestEvalFrame(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = paddle.to_tensor(2).astype('int')

    def tearDown(self):
        if False:
            return 10
        pass

    def test_eval_frame(self):
        if False:
            print('Hello World!')
        if version_info.major != 3 or (version_info.minor <= 8 or version_info.minor >= 12):
            return
        CustomCode = collections.namedtuple('CustomCode', ['code', 'disable_eval_frame'])

        def mul(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a * b
        code = CustomCode(mul.__code__, True)

        def callback(frame_obj):
            if False:
                return 10
            if frame_obj.f_code.co_name == 'add':
                return code
            return CustomCode(code=frame_obj.f_code, disable_eval_frame=True)

        def add(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a + b
        x = 1
        y = 2
        paddle.base.core.set_eval_frame(callback)
        assert add(x, y) == 2, 'should be 2'
        paddle.base.core.set_eval_frame(None)
        assert add(x, y) == 3, 'should be 3'
if __name__ == '__main__':
    unittest.main()