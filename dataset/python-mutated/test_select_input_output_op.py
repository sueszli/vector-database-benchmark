import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward
from paddle.base.executor import Executor
from paddle.base.framework import Program, program_guard
from paddle.static.nn.control_flow import select_input, select_output
paddle.enable_static()

class TestSplitMergeSelectedVarOps(unittest.TestCase):

    def test_forward_backward_list_output(self):
        if False:
            while True:
                i = 10
        for branch_num in range(2, 10):
            program = Program()
            with program_guard(program):
                x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
                x.stop_gradient = False
                mask = paddle.static.data(name='mask', shape=[-1, 1], dtype='int32')
                outputs = []
                for i in range(branch_num):
                    out = program.current_block().create_var(dtype='float32', shape=[2], type=core.VarDesc.VarType.LOD_TENSOR)
                    outputs.append(out)
                select_output(x, outputs, mask)
                y = select_input(outputs, mask)
                mean = paddle.mean(y)
                append_backward(mean)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = Executor(place)
            feed_x = np.asarray([1.3, -1.4]).astype(np.float32)
            for i in range(branch_num):
                feed_mask = np.asarray([i]).astype(np.int32)
                ret = exe.run(program, feed={'x': feed_x, 'mask': feed_mask}, fetch_list=[y.name, x.grad_name])
                x_grad = np.asarray([0.5, 0.5]).astype(np.float32)
                np.testing.assert_allclose(np.asarray(ret[0]), feed_x, rtol=1e-05)
                np.testing.assert_allclose(np.asarray(ret[1]), x_grad, rtol=1e-05)

class TestSelectInputOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        with program_guard(Program(), Program()):
            mask = paddle.static.data(name='mask', shape=[-1, 1], dtype='int32')
            in1 = paddle.static.data(name='in1', shape=[-1, 1], dtype='int32')

            def test_inputs_type():
                if False:
                    i = 10
                    return i + 15
                select_input(1, mask)
            self.assertRaises(TypeError, test_inputs_type)

            def test_mask_type():
                if False:
                    return 10
                select_input([in1], mask=1)
            self.assertRaises(TypeError, test_mask_type)

            def test_mask_dtype():
                if False:
                    while True:
                        i = 10
                mask = paddle.static.data(name='mask2', shape=[-1, 1], dtype='float32')
                select_input([in1], mask)
            self.assertRaises(TypeError, test_mask_dtype)

class TestSelectOutput_Error(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):
            in1 = paddle.static.data(name='in1', shape=[-1, 1], dtype='int32')
            mask_int32 = paddle.static.data(name='mask_int32', shape=[-1, 1], dtype='int32')
            mask_float32 = paddle.static.data(name='mask_float32', shape=[-1, 1], dtype='float32')
            out1 = paddle.static.data(name='out1', shape=[-1, 1], dtype='int32')

            def test_input_type():
                if False:
                    print('Hello World!')
                select_output(1, [out1], mask_int32)
            self.assertRaises(TypeError, test_input_type)

            def test_mask_type():
                if False:
                    for i in range(10):
                        print('nop')
                select_output(in1, [out1], mask=1)
            self.assertRaises(TypeError, test_mask_type)

            def test_mask_dtype():
                if False:
                    i = 10
                    return i + 15
                select_output(in1, [out1], mask=mask_float32)
            self.assertRaises(TypeError, test_mask_dtype)

            def test_outputs_type():
                if False:
                    while True:
                        i = 10
                select_output(in1, out1, mask=mask_int32)
            self.assertRaises(TypeError, test_outputs_type)
if __name__ == '__main__':
    unittest.main()