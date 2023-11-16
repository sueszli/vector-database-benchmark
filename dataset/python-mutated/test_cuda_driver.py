from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import host_to_device, device_to_host, driver, launch_kernel
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
ptx1 = '\n    .version 1.4\n    .target sm_10, map_f64_to_f32\n\n    .entry _Z10helloworldPi (\n    .param .u64 __cudaparm__Z10helloworldPi_A)\n    {\n    .reg .u32 %r<3>;\n    .reg .u64 %rd<6>;\n    .loc\t14\t4\t0\n$LDWbegin__Z10helloworldPi:\n    .loc\t14\t6\t0\n    cvt.s32.u16 \t%r1, %tid.x;\n    ld.param.u64 \t%rd1, [__cudaparm__Z10helloworldPi_A];\n    cvt.u64.u16 \t%rd2, %tid.x;\n    mul.lo.u64 \t%rd3, %rd2, 4;\n    add.u64 \t%rd4, %rd1, %rd3;\n    st.global.s32 \t[%rd4+0], %r1;\n    .loc\t14\t7\t0\n    exit;\n$LDWend__Z10helloworldPi:\n    } // _Z10helloworldPi\n'
ptx2 = '\n.version 3.0\n.target sm_20\n.address_size 64\n\n    .file\t1 "/tmp/tmpxft_000012c7_00000000-9_testcuda.cpp3.i"\n    .file\t2 "testcuda.cu"\n\n.entry _Z10helloworldPi(\n    .param .u64 _Z10helloworldPi_param_0\n)\n{\n    .reg .s32 \t%r<3>;\n    .reg .s64 \t%rl<5>;\n\n\n    ld.param.u64 \t%rl1, [_Z10helloworldPi_param_0];\n    cvta.to.global.u64 \t%rl2, %rl1;\n    .loc 2 6 1\n    mov.u32 \t%r1, %tid.x;\n    mul.wide.u32 \t%rl3, %r1, 4;\n    add.s64 \t%rl4, %rl2, %rl3;\n    st.global.u32 \t[%rl4], %r1;\n    .loc 2 7 2\n    ret;\n}\n'

@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestCudaDriver(CUDATestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.assertTrue(len(devices.gpus) > 0)
        self.context = devices.get_context()
        device = self.context.device
        (ccmajor, _) = device.compute_capability
        if ccmajor >= 2:
            self.ptx = ptx2
        else:
            self.ptx = ptx1

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        del self.context

    def test_cuda_driver_basic(self):
        if False:
            return 10
        module = self.context.create_module_ptx(self.ptx)
        function = module.get_function('_Z10helloworldPi')
        array = (c_int * 100)()
        memory = self.context.memalloc(sizeof(array))
        host_to_device(memory, array, sizeof(array))
        ptr = memory.device_ctypes_pointer
        stream = 0
        if _driver.USE_NV_BINDING:
            ptr = c_void_p(int(ptr))
            stream = _driver.binding.CUstream(stream)
        launch_kernel(function.handle, 1, 1, 1, 100, 1, 1, 0, stream, [ptr])
        device_to_host(array, memory, sizeof(array))
        for (i, v) in enumerate(array):
            self.assertEqual(i, v)
        module.unload()

    def test_cuda_driver_stream_operations(self):
        if False:
            i = 10
            return i + 15
        module = self.context.create_module_ptx(self.ptx)
        function = module.get_function('_Z10helloworldPi')
        array = (c_int * 100)()
        stream = self.context.create_stream()
        with stream.auto_synchronize():
            memory = self.context.memalloc(sizeof(array))
            host_to_device(memory, array, sizeof(array), stream=stream)
            ptr = memory.device_ctypes_pointer
            if _driver.USE_NV_BINDING:
                ptr = c_void_p(int(ptr))
            launch_kernel(function.handle, 1, 1, 1, 100, 1, 1, 0, stream.handle, [ptr])
        device_to_host(array, memory, sizeof(array), stream=stream)
        for (i, v) in enumerate(array):
            self.assertEqual(i, v)

    def test_cuda_driver_default_stream(self):
        if False:
            i = 10
            return i + 15
        ds = self.context.get_default_stream()
        self.assertIn('Default CUDA stream', repr(ds))
        self.assertEqual(0, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_legacy_default_stream(self):
        if False:
            for i in range(10):
                print('nop')
        ds = self.context.get_legacy_default_stream()
        self.assertIn('Legacy default CUDA stream', repr(ds))
        self.assertEqual(1, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_per_thread_default_stream(self):
        if False:
            return 10
        ds = self.context.get_per_thread_default_stream()
        self.assertIn('Per-thread default CUDA stream', repr(ds))
        self.assertEqual(2, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_stream(self):
        if False:
            while True:
                i = 10
        s = self.context.create_stream()
        self.assertIn('CUDA stream', repr(s))
        self.assertNotIn('Default', repr(s))
        self.assertNotIn('External', repr(s))
        self.assertNotEqual(0, int(s))
        self.assertTrue(s)
        self.assertFalse(s.external)

    def test_cuda_driver_external_stream(self):
        if False:
            return 10
        if _driver.USE_NV_BINDING:
            handle = driver.cuStreamCreate(0)
            ptr = int(handle)
        else:
            handle = drvapi.cu_stream()
            driver.cuStreamCreate(byref(handle), 0)
            ptr = handle.value
        s = self.context.create_external_stream(ptr)
        self.assertIn('External CUDA stream', repr(s))
        self.assertNotIn('efault', repr(s))
        self.assertEqual(ptr, int(s))
        self.assertTrue(s)
        self.assertTrue(s.external)

    def test_cuda_driver_occupancy(self):
        if False:
            return 10
        module = self.context.create_module_ptx(self.ptx)
        function = module.get_function('_Z10helloworldPi')
        value = self.context.get_active_blocks_per_multiprocessor(function, 128, 128)
        self.assertTrue(value > 0)

        def b2d(bs):
            if False:
                print('Hello World!')
            return bs
        (grid, block) = self.context.get_max_potential_block_size(function, b2d, 128, 128)
        self.assertTrue(grid > 0)
        self.assertTrue(block > 0)

class TestDevice(CUDATestCase):

    def test_device_get_uuid(self):
        if False:
            return 10
        h = '[0-9a-f]{%d}'
        h4 = h % 4
        h8 = h % 8
        h12 = h % 12
        uuid_format = f'^GPU-{h8}-{h4}-{h4}-{h4}-{h12}$'
        dev = devices.get_context().device
        self.assertRegex(dev.uuid, uuid_format)
if __name__ == '__main__':
    unittest.main()