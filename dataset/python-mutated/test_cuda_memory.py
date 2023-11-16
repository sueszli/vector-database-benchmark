import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim

@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestCudaMemory(ContextResettingTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.context = devices.get_context()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        del self.context
        super(TestCudaMemory, self).tearDown()

    def _template(self, obj):
        if False:
            while True:
                i = 10
        self.assertTrue(driver.is_device_memory(obj))
        driver.require_device_memory(obj)
        if driver.USE_NV_BINDING:
            expected_class = driver.binding.CUdeviceptr
        else:
            expected_class = drvapi.cu_device_ptr
        self.assertTrue(isinstance(obj.device_ctypes_pointer, expected_class))

    def test_device_memory(self):
        if False:
            while True:
                i = 10
        devmem = self.context.memalloc(1024)
        self._template(devmem)

    def test_device_view(self):
        if False:
            for i in range(10):
                print('nop')
        devmem = self.context.memalloc(1024)
        self._template(devmem.view(10))

    def test_host_alloc(self):
        if False:
            print('Hello World!')
        devmem = self.context.memhostalloc(1024, mapped=True)
        self._template(devmem)

    def test_pinned_memory(self):
        if False:
            for i in range(10):
                print('nop')
        ary = np.arange(10)
        devmem = self.context.mempin(ary, ary.ctypes.data, ary.size * ary.dtype.itemsize, mapped=True)
        self._template(devmem)

    def test_managed_memory(self):
        if False:
            return 10
        devmem = self.context.memallocmanaged(1024)
        self._template(devmem)

    def test_derived_pointer(self):
        if False:
            print('Hello World!')

        def handle_val(mem):
            if False:
                return 10
            if driver.USE_NV_BINDING:
                return int(mem.handle)
            else:
                return mem.handle.value

        def check(m, offset):
            if False:
                for i in range(10):
                    print('nop')
            v1 = m.view(offset)
            self.assertEqual(handle_val(v1.owner), handle_val(m))
            self.assertEqual(m.refct, 2)
            self.assertEqual(handle_val(v1) - offset, handle_val(v1.owner))
            v2 = v1.view(offset)
            self.assertEqual(handle_val(v2.owner), handle_val(m))
            self.assertEqual(handle_val(v2.owner), handle_val(m))
            self.assertEqual(handle_val(v2) - offset * 2, handle_val(v2.owner))
            self.assertEqual(m.refct, 3)
            del v2
            self.assertEqual(m.refct, 2)
            del v1
            self.assertEqual(m.refct, 1)
        m = self.context.memalloc(1024)
        check(m=m, offset=0)
        check(m=m, offset=1)

    def test_user_extension(self):
        if False:
            for i in range(10):
                print('nop')
        fake_ptr = ctypes.c_void_p(3735928559)
        dtor_invoked = [0]

        def dtor():
            if False:
                while True:
                    i = 10
            dtor_invoked[0] += 1
        ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr, size=40, finalizer=dtor)
        self.assertEqual(dtor_invoked[0], 0)
        del ptr
        self.assertEqual(dtor_invoked[0], 1)
        ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr, size=40, finalizer=dtor)
        owned = ptr.own()
        del owned
        self.assertEqual(dtor_invoked[0], 1)
        del ptr
        self.assertEqual(dtor_invoked[0], 2)

class TestCudaMemoryFunctions(ContextResettingTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.context = devices.get_context()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        del self.context
        super(TestCudaMemoryFunctions, self).tearDown()

    def test_memcpy(self):
        if False:
            for i in range(10):
                print('nop')
        hstary = np.arange(100, dtype=np.uint32)
        hstary2 = np.arange(100, dtype=np.uint32)
        sz = hstary.size * hstary.dtype.itemsize
        devary = self.context.memalloc(sz)
        driver.host_to_device(devary, hstary, sz)
        driver.device_to_host(hstary2, devary, sz)
        self.assertTrue(np.all(hstary == hstary2))

    def test_memset(self):
        if False:
            i = 10
            return i + 15
        dtype = np.dtype('uint32')
        n = 10
        sz = dtype.itemsize * 10
        devary = self.context.memalloc(sz)
        driver.device_memset(devary, 171, sz)
        hstary = np.empty(n, dtype=dtype)
        driver.device_to_host(hstary, devary, sz)
        hstary2 = np.array([2880154539] * n, dtype=np.dtype('uint32'))
        self.assertTrue(np.all(hstary == hstary2))

    def test_d2d(self):
        if False:
            return 10
        hst = np.arange(100, dtype=np.uint32)
        hst2 = np.empty_like(hst)
        sz = hst.size * hst.dtype.itemsize
        dev1 = self.context.memalloc(sz)
        dev2 = self.context.memalloc(sz)
        driver.host_to_device(dev1, hst, sz)
        driver.device_to_device(dev2, dev1, sz)
        driver.device_to_host(hst2, dev2, sz)
        self.assertTrue(np.all(hst == hst2))

@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestMVExtent(ContextResettingTestCase):

    def test_c_contiguous_array(self):
        if False:
            for i in range(10):
                print('nop')
        ary = np.arange(100)
        arysz = ary.dtype.itemsize * ary.size
        (s, e) = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_f_contiguous_array(self):
        if False:
            for i in range(10):
                print('nop')
        ary = np.asfortranarray(np.arange(100).reshape(2, 50))
        arysz = ary.dtype.itemsize * np.prod(ary.shape)
        (s, e) = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_single_element_array(self):
        if False:
            for i in range(10):
                print('nop')
        ary = np.asarray(np.uint32(1234))
        arysz = ary.dtype.itemsize
        (s, e) = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_ctypes_struct(self):
        if False:
            i = 10
            return i + 15

        class mystruct(ctypes.Structure):
            _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]
        data = mystruct(x=123, y=432)
        sz = driver.host_memory_size(data)
        self.assertTrue(ctypes.sizeof(data) == sz)

    def test_ctypes_double(self):
        if False:
            print('Hello World!')
        data = ctypes.c_double(1.234)
        sz = driver.host_memory_size(data)
        self.assertTrue(ctypes.sizeof(data) == sz)
if __name__ == '__main__':
    unittest.main()