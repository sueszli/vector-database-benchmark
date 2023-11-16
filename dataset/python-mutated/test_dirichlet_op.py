import unittest
import numpy as np
import scipy.stats
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle
from paddle.base import core
paddle.enable_static()

class TestDirichletOp(OpTest):
    no_need_check_grad = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'dirichlet'
        self.alpha = np.array((1.0, 2.0))
        self.sample_shape = (100000, 2)
        self.inputs = {'Alpha': np.broadcast_to(self.alpha, self.sample_shape)}
        self.attrs = {}
        self.outputs = {'Out': np.zeros(self.sample_shape)}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output_customized(self._hypothesis_testing)

    def _hypothesis_testing(self, outs):
        if False:
            print('Hello World!')
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(scipy.stats.kstest(outs[0][:, 0], scipy.stats.beta(a=self.alpha[0], b=self.alpha[1]).cdf)[0], 0.01)

class TestDirichletFP16Op(OpTest):
    no_need_check_grad = True

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'dirichlet'
        self.alpha = np.array((1.0, 2.0))
        self.sample_shape = (100000, 2)
        self.dtype = np.float16
        self.inputs = {'Alpha': np.broadcast_to(self.alpha, self.sample_shape).astype(self.dtype)}
        self.attrs = {}
        self.outputs = {'Out': np.zeros(self.sample_shape).astype(self.dtype)}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_customized(self._hypothesis_testing)

    def _hypothesis_testing(self, outs):
        if False:
            while True:
                i = 10
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(scipy.stats.kstest(outs[0][:, 0], scipy.stats.beta(a=self.alpha[0], b=self.alpha[1]).cdf)[0], 0.01)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestDirichletBF16Op(OpTest):
    no_need_check_grad = True

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'dirichlet'
        self.alpha = np.array((1.0, 2.0))
        self.sample_shape = (10000, 2)
        self.dtype = np.uint16
        self.np_dtype = np.float32
        self.inputs = {'Alpha': np.broadcast_to(self.alpha, self.sample_shape).astype(self.np_dtype)}
        self.attrs = {}
        self.outputs = {'Out': np.zeros(self.sample_shape).astype(self.np_dtype)}
        self.inputs['Alpha'] = convert_float_to_uint16(self.inputs['Alpha'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        if False:
            return 10
        self.check_output_with_place_customized(self._hypothesis_testing, place=core.CUDAPlace(0))

    def _hypothesis_testing(self, outs):
        if False:
            while True:
                i = 10
        outs = convert_uint16_to_float(outs)
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(scipy.stats.kstest(outs[0][:, 0], scipy.stats.beta(a=self.alpha[0], b=self.alpha[1]).cdf)[0], 0.3)
if __name__ == '__main__':
    unittest.main()