import unittest
import numpy as np
import paddle
paddle.enable_static()

class TestGetInputsOutputsInBlock(unittest.TestCase):

    def test_ordered(self):
        if False:
            while True:
                i = 10
        self._test_while_loop()
        self._test_cond()

    def _test_while_loop(self):
        if False:
            while True:
                i = 10
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.assign(np.array([1]))
            ten = paddle.assign(np.array([10]))

            def while_cond(i):
                if False:
                    return 10
                return i < ten

            def while_body(i):
                if False:
                    for i in range(10):
                        print('nop')
                one = paddle.assign(np.array([1]))
                i = i + one
                return [i]
            i = paddle.static.nn.while_loop(while_cond, while_body, [i])
        sub_block = main_program.block(1)
        (inner_inputs, inner_outputs) = paddle.utils.get_inputs_outputs_in_block(sub_block)
        self.assertTrue(inner_inputs == {'assign_0.tmp_0', 'assign_1.tmp_0'})
        self.assertTrue(inner_outputs == {'tmp_0', 'assign_0.tmp_0'})

    def _test_cond(self):
        if False:
            for i in range(10):
                print('nop')
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.zeros((1, 1))
            b = paddle.zeros((1, 1))
            c = a * b
            out = paddle.static.nn.cond(a < b, lambda : a + c, lambda : b * b)
        sub_block = main_program.block(1)
        (inner_inputs, inner_outputs) = paddle.utils.get_inputs_outputs_in_block(sub_block)
        self.assertTrue(inner_inputs == {'fill_constant_1.tmp_0', 'tmp_3'})
        self.assertTrue(inner_outputs == {'_generated_var_1'})
if __name__ == '__main__':
    unittest.main()