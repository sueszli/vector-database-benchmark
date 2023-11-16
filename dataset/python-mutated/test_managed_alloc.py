import numpy as np
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim, skip_on_arm
from numba.tests.support import linux_only

@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
@linux_only
@skip_on_arm('Managed Alloc support is experimental/untested on ARM')
class TestManagedAlloc(ContextResettingTestCase):

    def get_total_gpu_memory(self):
        if False:
            for i in range(10):
                print('nop')
        if USE_NV_BINDING:
            (free, total) = driver.cuMemGetInfo()
            return total
        else:
            free = c_size_t()
            total = c_size_t()
            driver.cuMemGetInfo(byref(free), byref(total))
            return total.value

    def skip_if_cc_major_lt(self, min_required, reason):
        if False:
            i = 10
            return i + 15
        '\n        Skip the current test if the compute capability of the device is\n        less than `min_required`.\n        '
        ctx = cuda.current_context()
        cc_major = ctx.device.compute_capability[0]
        if cc_major < min_required:
            self.skipTest(reason)

    def test_managed_alloc_driver_undersubscribe(self):
        if False:
            return 10
        msg = 'Managed memory unsupported prior to CC 3.0'
        self.skip_if_cc_major_lt(3, msg)
        self._test_managed_alloc_driver(0.5)

    @unittest.skip
    def test_managed_alloc_driver_oversubscribe(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Oversubscription of managed memory unsupported prior to CC 6.0'
        self.skip_if_cc_major_lt(6, msg)
        self._test_managed_alloc_driver(2.0)

    def test_managed_alloc_driver_host_attach(self):
        if False:
            return 10
        msg = 'Host attached managed memory is not accessible prior to CC 6.0'
        self.skip_if_cc_major_lt(6, msg)
        self._test_managed_alloc_driver(0.01, attach_global=False)

    def _test_managed_alloc_driver(self, memory_factor, attach_global=True):
        if False:
            while True:
                i = 10
        total_mem_size = self.get_total_gpu_memory()
        n_bytes = int(memory_factor * total_mem_size)
        ctx = cuda.current_context()
        mem = ctx.memallocmanaged(n_bytes, attach_global=attach_global)
        dtype = np.dtype(np.uint8)
        n_elems = n_bytes // dtype.itemsize
        ary = np.ndarray(shape=n_elems, dtype=dtype, buffer=mem)
        magic = 171
        device_memset(mem, magic, n_bytes)
        ctx.synchronize()
        self.assertTrue(np.all(ary == magic))

    def _test_managed_array(self, attach_global=True):
        if False:
            for i in range(10):
                print('nop')
        ary = cuda.managed_array(100, dtype=np.double)
        ary.fill(123.456)
        self.assertTrue(all(ary == 123.456))

        @cuda.jit('void(double[:])')
        def kernel(x):
            if False:
                while True:
                    i = 10
            i = cuda.grid(1)
            if i < x.shape[0]:
                x[i] = 1.0
        kernel[10, 10](ary)
        cuda.current_context().synchronize()
        self.assertTrue(all(ary == 1.0))

    def test_managed_array_attach_global(self):
        if False:
            i = 10
            return i + 15
        self._test_managed_array()

    def test_managed_array_attach_host(self):
        if False:
            return 10
        self._test_managed_array()
        msg = 'Host attached managed memory is not accessible prior to CC 6.0'
        self.skip_if_cc_major_lt(6, msg)
        self._test_managed_array(attach_global=False)
if __name__ == '__main__':
    unittest.main()