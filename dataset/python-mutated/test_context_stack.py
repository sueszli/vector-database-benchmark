import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver

class TestContextStack(CUDATestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        cuda.close()

    def test_gpus_current(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(cuda.gpus.current, None)
        with cuda.gpus[0]:
            self.assertEqual(int(cuda.gpus.current.id), 0)

    def test_gpus_len(self):
        if False:
            while True:
                i = 10
        self.assertGreater(len(cuda.gpus), 0)

    def test_gpus_iter(self):
        if False:
            return 10
        gpulist = list(cuda.gpus)
        self.assertGreater(len(gpulist), 0)

class TestContextAPI(CUDATestCase):

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        cuda.close()

    def test_context_memory(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            mem = cuda.current_context().get_memory_info()
        except NotImplementedError:
            self.skipTest('EMM Plugin does not implement get_memory_info()')
        self.assertIsInstance(mem.free, numbers.Number)
        self.assertEquals(mem.free, mem[0])
        self.assertIsInstance(mem.total, numbers.Number)
        self.assertEquals(mem.total, mem[1])
        self.assertLessEqual(mem.free, mem.total)

    @unittest.skipIf(len(cuda.gpus) < 2, 'need more than 1 gpus')
    @skip_on_cudasim('CUDA HW required')
    def test_forbidden_context_switch(self):
        if False:
            i = 10
            return i + 15

        @cuda.require_context
        def switch_gpu():
            if False:
                i = 10
                return i + 15
            with cuda.gpus[1]:
                pass
        with cuda.gpus[0]:
            with self.assertRaises(RuntimeError) as raises:
                switch_gpu()
            self.assertIn('Cannot switch CUDA-context.', str(raises.exception))

    @unittest.skipIf(len(cuda.gpus) < 2, 'need more than 1 gpus')
    def test_accepted_context_switch(self):
        if False:
            i = 10
            return i + 15

        def switch_gpu():
            if False:
                while True:
                    i = 10
            with cuda.gpus[1]:
                return cuda.current_context().device.id
        with cuda.gpus[0]:
            devid = switch_gpu()
        self.assertEqual(int(devid), 1)

@skip_on_cudasim('CUDA HW required')
class Test3rdPartyContext(CUDATestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        cuda.close()

    def test_attached_primary(self, extra_work=lambda : None):
        if False:
            i = 10
            return i + 15
        the_driver = driver.driver
        if driver.USE_NV_BINDING:
            dev = driver.binding.CUdevice(0)
            hctx = the_driver.cuDevicePrimaryCtxRetain(dev)
        else:
            dev = 0
            hctx = driver.drvapi.cu_context()
            the_driver.cuDevicePrimaryCtxRetain(byref(hctx), dev)
        try:
            ctx = driver.Context(weakref.proxy(self), hctx)
            ctx.push()
            my_ctx = cuda.current_context()
            if driver.USE_NV_BINDING:
                self.assertEqual(int(my_ctx.handle), int(ctx.handle))
            else:
                self.assertEqual(my_ctx.handle.value, ctx.handle.value)
            extra_work()
        finally:
            ctx.pop()
            the_driver.cuDevicePrimaryCtxRelease(dev)

    def test_attached_non_primary(self):
        if False:
            while True:
                i = 10
        the_driver = driver.driver
        if driver.USE_NV_BINDING:
            flags = 0
            dev = driver.binding.CUdevice(0)
            hctx = the_driver.cuCtxCreate(flags, dev)
        else:
            hctx = driver.drvapi.cu_context()
            the_driver.cuCtxCreate(byref(hctx), 0, 0)
        try:
            cuda.current_context()
        except RuntimeError as e:
            self.assertIn('Numba cannot operate on non-primary CUDA context ', str(e))
        else:
            self.fail('No RuntimeError raised')
        finally:
            the_driver.cuCtxDestroy(hctx)

    def test_cudajit_in_attached_primary_context(self):
        if False:
            i = 10
            return i + 15

        def do():
            if False:
                i = 10
                return i + 15
            from numba import cuda

            @cuda.jit
            def foo(a):
                if False:
                    for i in range(10):
                        print('nop')
                for i in range(a.size):
                    a[i] = i
            a = cuda.device_array(10)
            foo[1, 1](a)
            self.assertEqual(list(a.copy_to_host()), list(range(10)))
        self.test_attached_primary(do)
if __name__ == '__main__':
    unittest.main()