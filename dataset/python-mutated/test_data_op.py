import unittest
import paddle
from paddle.base.layer_helper import LayerHelper

def data():
    if False:
        for i in range(10):
            print('nop')
    helper = LayerHelper('data', **locals())
    out = helper.create_variable_for_type_inference('float32')
    helper.append_op(type='data', inputs={}, outputs={'out': out}, attrs={'shape': [1, 1], 'dtype': 0, 'place': 0, 'name': 'x'})
    return out

class TestPir(unittest.TestCase):

    def test_with_pir(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                out = data()
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()