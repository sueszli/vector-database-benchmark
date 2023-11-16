import unittest
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import TEST_NUMPY
from torch.testing._internal.common_cuda import TEST_NUMBA_CUDA, TEST_CUDA, TEST_MULTIGPU
import torch
if TEST_NUMPY:
    import numpy
if TEST_NUMBA_CUDA:
    import numba.cuda

class TestNumbaIntegration(common.TestCase):

    @unittest.skipIf(not TEST_NUMPY, 'No numpy')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    def test_cuda_array_interface(self):
        if False:
            while True:
                i = 10
        "torch.Tensor exposes __cuda_array_interface__ for cuda tensors.\n\n        An object t is considered a cuda-tensor if:\n            hasattr(t, '__cuda_array_interface__')\n\n        A cuda-tensor provides a tensor description dict:\n            shape: (integer, ...) Tensor shape.\n            strides: (integer, ...) Tensor strides, in bytes.\n            typestr: (str) A numpy-style typestr.\n            data: (int, boolean) A (data_ptr, read-only) tuple.\n            version: (int) Version 0\n\n        See:\n        https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html\n        "
        types = [torch.DoubleTensor, torch.FloatTensor, torch.HalfTensor, torch.LongTensor, torch.IntTensor, torch.ShortTensor, torch.CharTensor, torch.ByteTensor]
        dtypes = [numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8]
        for (tp, npt) in zip(types, dtypes):
            cput = tp(10)
            self.assertFalse(hasattr(cput, '__cuda_array_interface__'))
            self.assertRaises(AttributeError, lambda : cput.__cuda_array_interface__)
            if tp not in (torch.HalfTensor,):
                indices_t = torch.empty(1, cput.size(0), dtype=torch.long).clamp_(min=0)
                sparse_t = torch.sparse_coo_tensor(indices_t, cput)
                self.assertFalse(hasattr(sparse_t, '__cuda_array_interface__'))
                self.assertRaises(AttributeError, lambda : sparse_t.__cuda_array_interface__)
                sparse_cuda_t = torch.sparse_coo_tensor(indices_t, cput).cuda()
                self.assertFalse(hasattr(sparse_cuda_t, '__cuda_array_interface__'))
                self.assertRaises(AttributeError, lambda : sparse_cuda_t.__cuda_array_interface__)
            cudat = tp(10).cuda()
            self.assertTrue(hasattr(cudat, '__cuda_array_interface__'))
            ar_dict = cudat.__cuda_array_interface__
            self.assertEqual(set(ar_dict.keys()), {'shape', 'strides', 'typestr', 'data', 'version'})
            self.assertEqual(ar_dict['shape'], (10,))
            self.assertIs(ar_dict['strides'], None)
            self.assertEqual(ar_dict['typestr'], numpy.dtype(npt).newbyteorder('<').str)
            self.assertEqual(ar_dict['data'], (cudat.data_ptr(), False))
            self.assertEqual(ar_dict['version'], 2)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    @unittest.skipIf(not TEST_NUMBA_CUDA, 'No numba.cuda')
    def test_array_adaptor(self):
        if False:
            while True:
                i = 10
        'Torch __cuda_array_adaptor__ exposes tensor data to numba.cuda.'
        torch_dtypes = [torch.complex64, torch.complex128, torch.float16, torch.float32, torch.float64, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
        for dt in torch_dtypes:
            cput = torch.arange(10).to(dt)
            npt = cput.numpy()
            self.assertTrue(not numba.cuda.is_cuda_array(cput))
            with self.assertRaises(TypeError):
                numba.cuda.as_cuda_array(cput)
            cudat = cput.to(device='cuda')
            self.assertTrue(numba.cuda.is_cuda_array(cudat))
            numba_view = numba.cuda.as_cuda_array(cudat)
            self.assertIsInstance(numba_view, numba.cuda.devicearray.DeviceNDArray)
            self.assertEqual(numba_view.dtype, npt.dtype)
            self.assertEqual(numba_view.strides, npt.strides)
            self.assertEqual(numba_view.shape, cudat.shape)
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to('cuda'))
            cudat[:5] = 11
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to('cuda'))
            strided_cudat = cudat[::2]
            strided_npt = cput[::2].numpy()
            strided_numba_view = numba.cuda.as_cuda_array(strided_cudat)
            self.assertEqual(strided_numba_view.dtype, strided_npt.dtype)
            self.assertEqual(strided_numba_view.strides, strided_npt.strides)
            self.assertEqual(strided_numba_view.shape, strided_cudat.shape)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    @unittest.skipIf(not TEST_NUMBA_CUDA, 'No numba.cuda')
    def test_conversion_errors(self):
        if False:
            for i in range(10):
                print('nop')
        'Numba properly detects array interface for tensor.Tensor variants.'
        cput = torch.arange(100)
        self.assertFalse(numba.cuda.is_cuda_array(cput))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cput)
        sparset = torch.sparse_coo_tensor(cput[None, :], cput)
        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)
        sparse_cuda_t = sparset.cuda()
        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)
        cpu_gradt = torch.zeros(100).requires_grad_(True)
        self.assertFalse(numba.cuda.is_cuda_array(cpu_gradt))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cpu_gradt)
        cuda_gradt = torch.zeros(100).requires_grad_(True).cuda()
        with self.assertRaises(RuntimeError):
            numba.cuda.is_cuda_array(cuda_gradt)
        with self.assertRaises(RuntimeError):
            numba.cuda.as_cuda_array(cuda_gradt)

    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    @unittest.skipIf(not TEST_NUMBA_CUDA, 'No numba.cuda')
    @unittest.skipIf(not TEST_MULTIGPU, 'No multigpu')
    def test_active_device(self):
        if False:
            for i in range(10):
                print('nop')
        "'as_cuda_array' tensor device must match active numba context."
        cudat = torch.arange(10, device='cuda')
        self.assertEqual(cudat.device.index, 0)
        self.assertIsInstance(numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray)
        cudat = torch.arange(10, device=torch.device('cuda', 1))
        with self.assertRaises(numba.cuda.driver.CudaAPIError):
            numba.cuda.as_cuda_array(cudat)
        with numba.cuda.devices.gpus[cudat.device.index]:
            self.assertIsInstance(numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray)

    @unittest.skip('Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418')
    @unittest.skipIf(not TEST_NUMPY, 'No numpy')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    @unittest.skipIf(not TEST_NUMBA_CUDA, 'No numba.cuda')
    def test_from_cuda_array_interface(self):
        if False:
            for i in range(10):
                print('nop')
        'torch.as_tensor() and torch.tensor() supports the __cuda_array_interface__ protocol.\n\n        If an object exposes the __cuda_array_interface__, .as_tensor() and .tensor()\n        will use the exposed device memory.\n\n        See:\n        https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html\n        '
        dtypes = [numpy.complex64, numpy.complex128, numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8]
        for dtype in dtypes:
            numpy_arys = [numpy.arange(6).reshape(2, 3).astype(dtype), numpy.arange(6).reshape(2, 3).astype(dtype)[1:], numpy.arange(6).reshape(2, 3).astype(dtype)[:, None]]
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                torch_ary = torch.as_tensor(numba_ary, device='cuda')
                self.assertEqual(numba_ary.__cuda_array_interface__, torch_ary.__cuda_array_interface__)
                self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary, dtype=dtype))
                torch_ary += 42
                self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary, dtype=dtype))
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                torch_ary = torch.as_tensor(numba_ary, device='cpu')
                self.assertEqual(torch_ary.data.numpy(), numpy.asarray(numba_ary, dtype=dtype))
                torch_ary += 42
                self.assertEqual(torch_ary.data.numpy(), numpy.asarray(numba_ary, dtype=dtype) + 42)
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                torch_ary = torch.tensor(numba_ary, device='cuda')
                self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary, dtype=dtype))
                torch_ary += 42
                self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary, dtype=dtype) + 42)

    @unittest.skipIf(not TEST_NUMPY, 'No numpy')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    @unittest.skipIf(not TEST_NUMBA_CUDA, 'No numba.cuda')
    def test_from_cuda_array_interface_inferred_strides(self):
        if False:
            print('Hello World!')
        'torch.as_tensor(numba_ary) should have correct inferred (contiguous) strides'
        dtypes = [numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8]
        for dtype in dtypes:
            numpy_ary = numpy.arange(6).reshape(2, 3).astype(dtype)
            numba_ary = numba.cuda.to_device(numpy_ary)
            self.assertTrue(numba_ary.is_c_contiguous())
            torch_ary = torch.as_tensor(numba_ary, device='cuda')
            self.assertTrue(torch_ary.is_contiguous())

    @unittest.skip('Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418')
    @unittest.skipIf(not TEST_NUMPY, 'No numpy')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    @unittest.skipIf(not TEST_NUMBA_CUDA, 'No numba.cuda')
    def test_from_cuda_array_interface_lifetime(self):
        if False:
            return 10
        'torch.as_tensor(obj) tensor grabs a reference to obj so that the lifetime of obj exceeds the tensor'
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device='cuda')
        self.assertEqual(torch_ary.__cuda_array_interface__, numba_ary.__cuda_array_interface__)
        del numba_ary
        self.assertEqual(torch_ary.cpu().data.numpy(), numpy.arange(6))

    @unittest.skip('Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418')
    @unittest.skipIf(not TEST_NUMPY, 'No numpy')
    @unittest.skipIf(not TEST_CUDA, 'No cuda')
    @unittest.skipIf(not TEST_NUMBA_CUDA, 'No numba.cuda')
    @unittest.skipIf(not TEST_MULTIGPU, 'No multigpu')
    def test_from_cuda_array_interface_active_device(self):
        if False:
            for i in range(10):
                print('nop')
        'torch.as_tensor() tensor device must match active numba context.'
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device='cuda')
        self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary))
        self.assertEqual(torch_ary.__cuda_array_interface__, numba_ary.__cuda_array_interface__)
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device=torch.device('cuda', 1))
        self.assertEqual(torch_ary.get_device(), 1)
        self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary))
        if1 = torch_ary.__cuda_array_interface__
        if2 = numba_ary.__cuda_array_interface__
        self.assertNotEqual(if1['data'], if2['data'])
        del if1['data']
        del if2['data']
        self.assertEqual(if1, if2)
if __name__ == '__main__':
    common.run_tests()