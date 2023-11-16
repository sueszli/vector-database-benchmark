import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
if not config.ENABLE_CUDASIM:

    class DeviceOnlyEMMPlugin(cuda.HostOnlyCUDAMemoryManager):
        """
        Dummy EMM Plugin implementation for testing. It memorises which plugin
        API methods have been called so that the tests can check that Numba
        called into the plugin as expected.
        """

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(*args, **kwargs)
            self.allocations = {}
            self.count = 0
            self.initialized = False
            self.memalloc_called = False
            self.reset_called = False
            self.get_memory_info_called = False
            self.get_ipc_handle_called = False

        def memalloc(self, size):
            if False:
                print('Hello World!')
            if not self.initialized:
                raise RuntimeError('memalloc called before initialize')
            self.memalloc_called = True
            self.count += 1
            alloc_count = self.count
            self.allocations[alloc_count] = size
            finalizer_allocs = self.allocations

            def finalizer():
                if False:
                    print('Hello World!')
                del finalizer_allocs[alloc_count]
            ctx = weakref.proxy(self.context)
            ptr = ctypes.c_void_p(alloc_count)
            return cuda.cudadrv.driver.AutoFreePointer(ctx, ptr, size, finalizer=finalizer)

        def initialize(self):
            if False:
                i = 10
                return i + 15
            self.initialized = True

        def reset(self):
            if False:
                while True:
                    i = 10
            self.reset_called = True

        def get_memory_info(self):
            if False:
                print('Hello World!')
            self.get_memory_info_called = True
            return cuda.MemoryInfo(free=32, total=64)

        def get_ipc_handle(self, memory):
            if False:
                return 10
            self.get_ipc_handle_called = True
            return 'Dummy IPC handle for alloc %s' % memory.device_pointer.value

        @property
        def interface_version(self):
            if False:
                i = 10
                return i + 15
            return 1

    class BadVersionEMMPlugin(DeviceOnlyEMMPlugin):
        """A plugin that claims to implement a different interface version"""

        @property
        def interface_version(self):
            if False:
                return 10
            return 2

@skip_on_cudasim('EMM Plugins not supported on CUDA simulator')
class TestDeviceOnlyEMMPlugin(CUDATestCase):
    """
    Tests that the API of an EMM Plugin that implements device allocations
    only is used correctly by Numba.
    """

    def setUp(self):
        if False:
            return 10
        super().setUp()
        cuda.close()
        cuda.set_memory_manager(DeviceOnlyEMMPlugin)

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        cuda.close()
        cuda.cudadrv.driver._memory_manager = None

    def test_memalloc(self):
        if False:
            print('Hello World!')
        mgr = cuda.current_context().memory_manager
        arr_1 = np.arange(10)
        d_arr_1 = cuda.device_array_like(arr_1)
        self.assertTrue(mgr.memalloc_called)
        self.assertEqual(mgr.count, 1)
        self.assertEqual(mgr.allocations[1], arr_1.nbytes)
        arr_2 = np.arange(5)
        d_arr_2 = cuda.device_array_like(arr_2)
        self.assertEqual(mgr.count, 2)
        self.assertEqual(mgr.allocations[2], arr_2.nbytes)
        del d_arr_1
        self.assertNotIn(1, mgr.allocations)
        self.assertIn(2, mgr.allocations)
        del d_arr_2
        self.assertNotIn(2, mgr.allocations)

    def test_initialized_in_context(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(cuda.current_context().memory_manager.initialized)

    def test_reset(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = cuda.current_context()
        ctx.reset()
        self.assertTrue(ctx.memory_manager.reset_called)

    def test_get_memory_info(self):
        if False:
            while True:
                i = 10
        ctx = cuda.current_context()
        meminfo = ctx.get_memory_info()
        self.assertTrue(ctx.memory_manager.get_memory_info_called)
        self.assertEqual(meminfo.free, 32)
        self.assertEqual(meminfo.total, 64)

    @linux_only
    def test_get_ipc_handle(self):
        if False:
            for i in range(10):
                print('nop')
        arr = np.arange(2)
        d_arr = cuda.device_array_like(arr)
        ipch = d_arr.get_ipc_handle()
        ctx = cuda.current_context()
        self.assertTrue(ctx.memory_manager.get_ipc_handle_called)
        self.assertIn('Dummy IPC handle for alloc 1', ipch._ipc_handle)

@skip_on_cudasim('EMM Plugins not supported on CUDA simulator')
class TestBadEMMPluginVersion(CUDATestCase):
    """
    Ensure that Numba rejects EMM Plugins with incompatible version
    numbers.
    """

    def test_bad_plugin_version(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(RuntimeError) as raises:
            cuda.set_memory_manager(BadVersionEMMPlugin)
        self.assertIn('version 1 required', str(raises.exception))
if __name__ == '__main__':
    unittest.main()