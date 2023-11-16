import unittest
import numpy
import paddle
from paddle import base
from paddle.base import core
from paddle.base.executor import Executor

class TestTrunctedGaussianRandomOp(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'truncated_gaussian_random'
        self.inputs = {}
        self.attrs = {'shape': [10000], 'mean': 0.0, 'std': 1.0, 'seed': 10}
        self.outputs = ['Out']

    def test_cpu(self):
        if False:
            i = 10
            return i + 15
        self._gaussian_random_test(place=base.CPUPlace(), dtype=core.VarDesc.VarType.FP32)
        self._gaussian_random_test(place=base.CPUPlace(), dtype=core.VarDesc.VarType.FP64)
        self._gaussian_random_test_eager(place=base.CPUPlace(), dtype=core.VarDesc.VarType.FP32)
        self._gaussian_random_test_eager(place=base.CPUPlace(), dtype=core.VarDesc.VarType.FP64)

    def test_gpu(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            self._gaussian_random_test(place=base.CUDAPlace(0), dtype=core.VarDesc.VarType.FP32)
            self._gaussian_random_test(place=base.CUDAPlace(0), dtype=core.VarDesc.VarType.FP64)
            self._gaussian_random_test_eager(place=base.CUDAPlace(0), dtype=core.VarDesc.VarType.FP32)
            self._gaussian_random_test_eager(place=base.CUDAPlace(0), dtype=core.VarDesc.VarType.FP64)

    def _gaussian_random_test(self, place, dtype):
        if False:
            while True:
                i = 10
        program = base.Program()
        block = program.global_block()
        vout = block.create_var(name='Out')
        op = block.append_op(type=self.op_type, outputs={'Out': vout}, attrs={**self.attrs, 'dtype': dtype})
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)
        fetch_list = []
        for var_name in self.outputs:
            fetch_list.append(block.var(var_name))
        exe = Executor(place)
        outs = exe.run(program, fetch_list=fetch_list)
        tensor = outs[0]
        self.assertAlmostEqual(numpy.mean(tensor), 0.0, delta=0.1)
        self.assertAlmostEqual(numpy.var(tensor), 0.773, delta=0.1)

    def _gaussian_random_test_eager(self, place, dtype):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(place):
            out = paddle._C_ops.truncated_gaussian_random(self.attrs['shape'], self.attrs['mean'], self.attrs['std'], self.attrs['seed'], dtype, place)
            self.assertAlmostEqual(numpy.mean(out.numpy()), 0.0, delta=0.1)
            self.assertAlmostEqual(numpy.var(out.numpy()), 0.773, delta=0.1)
if __name__ == '__main__':
    unittest.main()