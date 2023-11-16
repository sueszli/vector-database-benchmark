import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from ctypes import c_size_t, c_uint64, sizeof
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
is64bit = sizeof(c_size_t) == sizeof(c_uint64)

@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestNvvmDriver(unittest.TestCase):

    def get_nvvmir(self):
        if False:
            return 10
        versions = NVVM().get_ir_version()
        metadata = metadata_nvvm70 % versions
        data_layout = NVVM().data_layout
        return nvvmir_generic.format(data_layout=data_layout, metadata=metadata)

    def test_nvvm_compile_simple(self):
        if False:
            while True:
                i = 10
        nvvmir = self.get_nvvmir()
        ptx = nvvm.llvm_to_ptx(nvvmir).decode('utf8')
        self.assertTrue('simple' in ptx)
        self.assertTrue('ave' in ptx)

    def test_nvvm_compile_nullary_option(self):
        if False:
            i = 10
            return i + 15
        if runtime.get_version() < (11, 5):
            self.skipTest('-gen-lto unavailable in this toolkit version')
        nvvmir = self.get_nvvmir()
        ltoir = nvvm.llvm_to_ptx(nvvmir, opt=3, gen_lto=None, arch='compute_52')
        self.assertEqual(ltoir[:4], b'\xedCN\x7f')

    def test_nvvm_bad_option(self):
        if False:
            while True:
                i = 10
        msg = '-made-up-option=2 is an unsupported option'
        with self.assertRaisesRegex(NvvmError, msg):
            nvvm.llvm_to_ptx('', made_up_option=2)

    def test_nvvm_from_llvm(self):
        if False:
            for i in range(10):
                print('nop')
        m = ir.Module('test_nvvm_from_llvm')
        m.triple = 'nvptx64-nvidia-cuda'
        nvvm.add_ir_version(m)
        fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
        kernel = ir.Function(m, fty, name='mycudakernel')
        bldr = ir.IRBuilder(kernel.append_basic_block('entry'))
        bldr.ret_void()
        nvvm.set_cuda_kernel(kernel)
        m.data_layout = NVVM().data_layout
        ptx = nvvm.llvm_to_ptx(str(m)).decode('utf8')
        self.assertTrue('mycudakernel' in ptx)
        if is64bit:
            self.assertTrue('.address_size 64' in ptx)
        else:
            self.assertTrue('.address_size 32' in ptx)

    def test_nvvm_ir_verify_fail(self):
        if False:
            print('Hello World!')
        m = ir.Module('test_bad_ir')
        m.triple = 'unknown-unknown-unknown'
        m.data_layout = NVVM().data_layout
        nvvm.add_ir_version(m)
        with self.assertRaisesRegex(NvvmError, 'Invalid target triple'):
            nvvm.llvm_to_ptx(str(m))

    def _test_nvvm_support(self, arch):
        if False:
            i = 10
            return i + 15
        compute_xx = 'compute_{0}{1}'.format(*arch)
        nvvmir = self.get_nvvmir()
        ptx = nvvm.llvm_to_ptx(nvvmir, arch=compute_xx, ftz=1, prec_sqrt=0, prec_div=0).decode('utf8')
        self.assertIn('.target sm_{0}{1}'.format(*arch), ptx)
        self.assertIn('simple', ptx)
        self.assertIn('ave', ptx)

    def test_nvvm_support(self):
        if False:
            while True:
                i = 10
        'Test supported CC by NVVM\n        '
        for arch in nvvm.get_supported_ccs():
            self._test_nvvm_support(arch=arch)

    def test_nvvm_warning(self):
        if False:
            return 10
        m = ir.Module('test_nvvm_warning')
        m.triple = 'nvptx64-nvidia-cuda'
        m.data_layout = NVVM().data_layout
        nvvm.add_ir_version(m)
        fty = ir.FunctionType(ir.VoidType(), [])
        kernel = ir.Function(m, fty, name='inlinekernel')
        builder = ir.IRBuilder(kernel.append_basic_block('entry'))
        builder.ret_void()
        nvvm.set_cuda_kernel(kernel)
        kernel.attributes.add('noinline')
        with warnings.catch_warnings(record=True) as w:
            nvvm.llvm_to_ptx(str(m))
        self.assertEqual(len(w), 1)
        self.assertIn('overriding noinline attribute', str(w[0]))

    @unittest.skipIf(True, 'No new CC unknown to NVVM yet')
    def test_nvvm_future_support(self):
        if False:
            print('Hello World!')
        'Test unsupported CC to help track the feature support\n        '
        future_archs = []
        for arch in future_archs:
            pat = '-arch=compute_{0}{1}'.format(*arch)
            with self.assertRaises(NvvmError) as raises:
                self._test_nvvm_support(arch=arch)
            self.assertIn(pat, raises.msg)

@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestArchOption(unittest.TestCase):

    def test_get_arch_option(self):
        if False:
            return 10
        self.assertEqual(nvvm.get_arch_option(5, 3), 'compute_53')
        self.assertEqual(nvvm.get_arch_option(7, 5), 'compute_75')
        self.assertEqual(nvvm.get_arch_option(7, 7), 'compute_75')
        supported_cc = nvvm.get_supported_ccs()
        for arch in supported_cc:
            self.assertEqual(nvvm.get_arch_option(*arch), 'compute_%d%d' % arch)
        self.assertEqual(nvvm.get_arch_option(1000, 0), 'compute_%d%d' % supported_cc[-1])

@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestLibDevice(unittest.TestCase):

    def test_libdevice_load(self):
        if False:
            print('Hello World!')
        libdevice = LibDevice()
        self.assertEqual(libdevice.bc[:4], b'BC\xc0\xde')
nvvmir_generic = 'target triple="nvptx64-nvidia-cuda"\ntarget datalayout = "{data_layout}"\n\ndefine i32 @ave(i32 %a, i32 %b) {{\nentry:\n%add = add nsw i32 %a, %b\n%div = sdiv i32 %add, 2\nret i32 %div\n}}\n\ndefine void @simple(i32* %data) {{\nentry:\n%0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n%1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n%mul = mul i32 %0, %1\n%2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n%add = add i32 %mul, %2\n%call = call i32 @ave(i32 %add, i32 %add)\n%idxprom = sext i32 %add to i64\n%arrayidx = getelementptr inbounds i32, i32* %data, i64 %idxprom\nstore i32 %call, i32* %arrayidx, align 4\nret void\n}}\n\ndeclare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone\n\ndeclare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone\n\ndeclare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone\n\n{metadata}\n'
metadata_nvvm70 = '\n!nvvmir.version = !{!1}\n!1 = !{i32 %s, i32 %s, i32 %s, i32 %s}\n\n!nvvm.annotations = !{!2}\n!2 = !{void (i32*)* @simple, !"kernel", i32 1}\n'
metadata_nvvm34 = '\n!nvvm.annotations = !{!1}\n!1 = metadata !{void (i32*)* @simple, metadata !"kernel", i32 1}\n'
if __name__ == '__main__':
    unittest.main()