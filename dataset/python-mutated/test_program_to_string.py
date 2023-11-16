import unittest
import paddle
from paddle import base

class TestProgram(unittest.TestCase):

    def test_program_to_string(self):
        if False:
            i = 10
            return i + 15
        prog = base.default_main_program()
        a = paddle.static.data(name='X', shape=[2, 3], dtype='float32')
        c = paddle.static.nn.fc(a, size=3)
        prog_string = prog.to_string(throw_on_error=True, with_details=False)
        prog_string_with_details = prog.to_string(throw_on_error=False, with_details=True)
        assert prog_string is not None
        assert len(prog_string_with_details) > len(prog_string)
if __name__ == '__main__':
    unittest.main()