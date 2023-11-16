import os
import shutil
import sys
import unittest
import warnings
import re
import tempfile
import subprocess
import glob
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_cuda import TEST_CUDNN, TEST_CUDA
import torch
import torch.backends.cudnn
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
from torch.testing._internal.common_utils import gradcheck
import torch.multiprocessing as mp
from torch.utils.cpp_extension import _TORCH_PATH, remove_extension_h_precompiler_headers, get_cxx_compiler, check_compiler_is_gcc
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and (ROCM_HOME is not None)
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None
TEST_MPS = torch.backends.mps.is_available()
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')

def remove_build_path():
    if False:
        print('Hello World!')
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        if IS_WINDOWS:
            subprocess.run(['rm', '-rf', default_build_root], stdout=subprocess.PIPE)
        else:
            shutil.rmtree(default_build_root)

class TestCppExtensionJIT(common.TestCase):
    """Tests just-in-time cpp extensions.
    Don't confuse this with the PyTorch JIT (aka TorchScript).
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        remove_build_path()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        remove_build_path()

    def test_jit_compile_extension(self):
        if False:
            i = 10
            return i + 15
        module = torch.utils.cpp_extension.load(name='jit_extension', sources=['cpp_extensions/jit_extension.cpp', 'cpp_extensions/jit_extension2.cpp'], extra_include_paths=['cpp_extensions'], extra_cflags=['-g'], verbose=True)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())
        z = module.exp_add(x, y)
        self.assertEqual(z, x.exp() + y.exp())
        doubler = module.Doubler(2, 2)
        self.assertIsNone(doubler.get().grad)
        self.assertEqual(doubler.get().sum(), 4)
        self.assertEqual(doubler.forward().sum(), 8)

    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), 'CUDA not found')
    def test_jit_cuda_extension(self):
        if False:
            return 10
        module = torch.utils.cpp_extension.load(name='torch_test_cuda_extension', sources=['cpp_extensions/cuda_extension.cpp', 'cpp_extensions/cuda_extension.cu'], extra_cuda_cflags=['-O2'], verbose=True, keep_intermediates=False)
        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)
        z = module.sigmoid_add(x, y).cpu()
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not TEST_MPS, 'MPS not found')
    def test_mps_extension(self):
        if False:
            for i in range(10):
                print('nop')
        module = torch.utils.cpp_extension.load(name='torch_test_mps_extension', sources=['cpp_extensions/mps_extension.mm'], verbose=True, keep_intermediates=False)
        tensor_length = 100000
        x = torch.randn(tensor_length, device='cpu', dtype=torch.float32)
        y = torch.randn(tensor_length, device='cpu', dtype=torch.float32)
        cpu_output = module.get_cpu_add_output(x, y)
        mps_output = module.get_mps_add_output(x.to('mps'), y.to('mps'))
        self.assertEqual(cpu_output, mps_output.to('cpu'))

    def _run_jit_cuda_archflags(self, flags, expected):
        if False:
            print('Hello World!')

        def _check_cuobjdump_output(expected_values, is_ptx=False):
            if False:
                while True:
                    i = 10
            elf_or_ptx = '--list-ptx' if is_ptx else '--list-elf'
            lib_ext = '.pyd' if IS_WINDOWS else '.so'
            ext_filename = glob.glob(os.path.join(temp_dir, 'cudaext_archflag*' + lib_ext))[0]
            command = ['cuobjdump', elf_or_ptx, ext_filename]
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output, err) = p.communicate()
            output = output.decode('ascii')
            err = err.decode('ascii')
            if not p.returncode == 0 or not err == '':
                raise AssertionError(f'Flags: {flags}\nReturncode: {p.returncode}\nStderr: {err}\nOutput: {output} ')
            actual_arches = sorted(re.findall('sm_\\d\\d', output))
            expected_arches = sorted(['sm_' + xx for xx in expected_values])
            self.assertEqual(actual_arches, expected_arches, msg=f'Flags: {flags},  Actual: {actual_arches},  Expected: {expected_arches}\nStderr: {err}\nOutput: {output}')
        temp_dir = tempfile.mkdtemp()
        old_envvar = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
        try:
            os.environ['TORCH_CUDA_ARCH_LIST'] = flags
            params = {'name': 'cudaext_archflags', 'sources': ['cpp_extensions/cuda_extension.cpp', 'cpp_extensions/cuda_extension.cu'], 'extra_cuda_cflags': ['-O2'], 'verbose': True, 'build_directory': temp_dir}
            if IS_WINDOWS:
                p = mp.Process(target=torch.utils.cpp_extension.load, kwargs=params)
                p.start()
                p.join()
            else:
                torch.utils.cpp_extension.load(**params)
            _check_cuobjdump_output(expected[0])
            if expected[1] is not None:
                _check_cuobjdump_output(expected[1], is_ptx=True)
        finally:
            if IS_WINDOWS:
                subprocess.run(['rm', '-rf', temp_dir], stdout=subprocess.PIPE)
            else:
                shutil.rmtree(temp_dir)
            if old_envvar is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST')
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = old_envvar

    @unittest.skipIf(not TEST_CUDA, 'CUDA not found')
    @unittest.skipIf(TEST_ROCM, 'disabled on rocm')
    def test_jit_cuda_archflags(self):
        if False:
            i = 10
            return i + 15
        n = torch.cuda.device_count()
        capabilities = {torch.cuda.get_device_capability(i) for i in range(n)}
        archflags = {'': ([f'{capability[0]}{capability[1]}' for capability in capabilities], None), 'Maxwell+Tegra;6.1': (['53', '61'], None), 'Volta': (['70'], ['70'])}
        if int(torch.version.cuda.split('.')[0]) >= 10:
            archflags['7.5+PTX'] = (['75'], ['75'])
            archflags['5.0;6.0+PTX;7.0;7.5'] = (['50', '60', '70', '75'], ['60'])
        if int(torch.version.cuda.split('.')[0]) < 12:
            archflags['Pascal 3.5'] = (['35', '60', '61'], None)
        for (flags, expected) in archflags.items():
            self._run_jit_cuda_archflags(flags, expected)

    @unittest.skipIf(not TEST_CUDNN, 'CuDNN not found')
    @unittest.skipIf(TEST_ROCM, 'Not supported on ROCm')
    def test_jit_cudnn_extension(self):
        if False:
            print('Hello World!')
        if IS_WINDOWS:
            extra_ldflags = ['cudnn.lib']
        else:
            extra_ldflags = ['-lcudnn']
        module = torch.utils.cpp_extension.load(name='torch_test_cudnn_extension', sources=['cpp_extensions/cudnn_extension.cpp'], extra_ldflags=extra_ldflags, verbose=True, with_cuda=True)
        x = torch.randn(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)
        module.cudnn_relu(x, y)
        self.assertEqual(torch.nn.functional.relu(x), y)
        with self.assertRaisesRegex(RuntimeError, 'same size'):
            y_incorrect = torch.zeros(20, device='cuda', dtype=torch.float32)
            module.cudnn_relu(x, y_incorrect)

    def test_inline_jit_compile_extension_with_functions_as_list(self):
        if False:
            print('Hello World!')
        cpp_source = '\n        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {\n          return x.tanh() + y.tanh();\n        }\n        '
        module = torch.utils.cpp_extension.load_inline(name='inline_jit_extension_with_functions_list', cpp_sources=cpp_source, functions='tanh_add', verbose=True)
        self.assertEqual(module.tanh_add.__doc__.split('\n')[2], 'tanh_add')
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_inline_jit_compile_extension_with_functions_as_dict(self):
        if False:
            for i in range(10):
                print('nop')
        cpp_source = '\n        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {\n          return x.tanh() + y.tanh();\n        }\n        '
        module = torch.utils.cpp_extension.load_inline(name='inline_jit_extension_with_functions_dict', cpp_sources=cpp_source, functions={'tanh_add': 'Tanh and then sum :D'}, verbose=True)
        self.assertEqual(module.tanh_add.__doc__.split('\n')[2], 'Tanh and then sum :D')

    def test_inline_jit_compile_extension_multiple_sources_and_no_functions(self):
        if False:
            for i in range(10):
                print('nop')
        cpp_source1 = '\n        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y) {\n          return x.sin() + y.sin();\n        }\n        '
        cpp_source2 = '\n        #include <torch/extension.h>\n        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y);\n        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n          m.def("sin_add", &sin_add, "sin(x) + sin(y)");\n        }\n        '
        module = torch.utils.cpp_extension.load_inline(name='inline_jit_extension', cpp_sources=[cpp_source1, cpp_source2], verbose=True)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = module.sin_add(x, y)
        self.assertEqual(z, x.sin() + y.sin())

    @unittest.skip('Temporarily disabled')
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), 'CUDA not found')
    def test_inline_jit_compile_extension_cuda(self):
        if False:
            return 10
        cuda_source = '\n        __global__ void cos_add_kernel(\n            const float* __restrict__ x,\n            const float* __restrict__ y,\n            float* __restrict__ output,\n            const int size) {\n          const auto index = blockIdx.x * blockDim.x + threadIdx.x;\n          if (index < size) {\n            output[index] = __cosf(x[index]) + __cosf(y[index]);\n          }\n        }\n\n        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {\n          auto output = torch::zeros_like(x);\n          const int threads = 1024;\n          const int blocks = (output.numel() + threads - 1) / threads;\n          cos_add_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());\n          return output;\n        }\n        '
        cpp_source = 'torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);'
        module = torch.utils.cpp_extension.load_inline(name='inline_jit_extension_cuda', cpp_sources=cpp_source, cuda_sources=cuda_source, functions=['cos_add'], verbose=True)
        self.assertEqual(module.cos_add.__doc__.split('\n')[2], 'cos_add')
        x = torch.randn(4, 4, device='cuda', dtype=torch.float32)
        y = torch.randn(4, 4, device='cuda', dtype=torch.float32)
        z = module.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    @unittest.skip('Temporarily disabled')
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), 'CUDA not found')
    def test_inline_jit_compile_custom_op_cuda(self):
        if False:
            i = 10
            return i + 15
        cuda_source = '\n        __global__ void cos_add_kernel(\n            const float* __restrict__ x,\n            const float* __restrict__ y,\n            float* __restrict__ output,\n            const int size) {\n          const auto index = blockIdx.x * blockDim.x + threadIdx.x;\n          if (index < size) {\n            output[index] = __cosf(x[index]) + __cosf(y[index]);\n          }\n        }\n\n        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {\n          auto output = torch::zeros_like(x);\n          const int threads = 1024;\n          const int blocks = (output.numel() + threads - 1) / threads;\n          cos_add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());\n          return output;\n        }\n        '
        cpp_source = '\n           #include <torch/library.h>\n           torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);\n\n           TORCH_LIBRARY(inline_jit_extension_custom_op_cuda, m) {\n             m.def("cos_add", cos_add);\n           }\n        '
        torch.utils.cpp_extension.load_inline(name='inline_jit_extension_custom_op_cuda', cpp_sources=cpp_source, cuda_sources=cuda_source, verbose=True, is_python_module=False)
        x = torch.randn(4, 4, device='cuda', dtype=torch.float32)
        y = torch.randn(4, 4, device='cuda', dtype=torch.float32)
        z = torch.ops.inline_jit_extension_custom_op_cuda.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    def test_inline_jit_compile_extension_throws_when_functions_is_bad(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            torch.utils.cpp_extension.load_inline(name='invalid_jit_extension', cpp_sources='', functions=5)

    def test_lenient_flag_handling_in_jit_extensions(self):
        if False:
            while True:
                i = 10
        cpp_source = '\n        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {\n          return x.tanh() + y.tanh();\n        }\n        '
        module = torch.utils.cpp_extension.load_inline(name='lenient_flag_handling_extension', cpp_sources=cpp_source, functions='tanh_add', extra_cflags=['-g\n\n', '-O0 -Wall'], extra_include_paths=['       cpp_extensions\n'], verbose=True)
        x = torch.zeros(100, dtype=torch.float32)
        y = torch.zeros(100, dtype=torch.float32)
        z = module.tanh_add(x, y).cpu()
        self.assertEqual(z, x.tanh() + y.tanh())

    @unittest.skip('Temporarily disabled')
    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), 'CUDA not found')
    def test_half_support(self):
        if False:
            print('Hello World!')
        '\n        Checks for an issue with operator< ambiguity for half when certain\n        THC headers are included.\n\n        See https://github.com/pytorch/pytorch/pull/10301#issuecomment-416773333\n        for the corresponding issue.\n        '
        cuda_source = '\n        template<typename T, typename U>\n        __global__ void half_test_kernel(const T* input, U* output) {\n            if (input[0] < input[1] || input[0] >= input[1]) {\n                output[0] = 123;\n            }\n        }\n\n        torch::Tensor half_test(torch::Tensor input) {\n            auto output = torch::empty(1, input.options().dtype(torch::kFloat));\n            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "half_test", [&] {\n                half_test_kernel<scalar_t><<<1, 1>>>(\n                    input.data<scalar_t>(),\n                    output.data<float>());\n            });\n            return output;\n        }\n        '
        module = torch.utils.cpp_extension.load_inline(name='half_test_extension', cpp_sources='torch::Tensor half_test(torch::Tensor input);', cuda_sources=cuda_source, functions=['half_test'], verbose=True)
        x = torch.randn(3, device='cuda', dtype=torch.half)
        result = module.half_test(x)
        self.assertEqual(result[0], 123)

    def test_reload_jit_extension(self):
        if False:
            i = 10
            return i + 15

        def compile(code):
            if False:
                print('Hello World!')
            return torch.utils.cpp_extension.load_inline(name='reloaded_jit_extension', cpp_sources=code, functions='f', verbose=True)
        module = compile('int f() { return 123; }')
        self.assertEqual(module.f(), 123)
        module = compile('int f() { return 456; }')
        self.assertEqual(module.f(), 456)
        module = compile('int f() { return 456; }')
        self.assertEqual(module.f(), 456)
        module = compile('int f() { return 789; }')
        self.assertEqual(module.f(), 789)

    def test_cpp_frontend_module_has_same_output_as_python(self, dtype=torch.double):
        if False:
            for i in range(10):
                print('nop')
        extension = torch.utils.cpp_extension.load(name='cpp_frontend_extension', sources='cpp_extensions/cpp_frontend_extension.cpp', verbose=True)
        input = torch.randn(2, 5, dtype=dtype)
        cpp_linear = extension.Net(5, 2)
        cpp_linear.to(dtype)
        python_linear = torch.nn.Linear(5, 2).to(dtype)
        cpp_parameters = dict(cpp_linear.named_parameters())
        with torch.no_grad():
            python_linear.weight.copy_(cpp_parameters['fc.weight'])
            python_linear.bias.copy_(cpp_parameters['fc.bias'])
        cpp_output = cpp_linear.forward(input)
        python_output = python_linear(input)
        self.assertEqual(cpp_output, python_output)
        cpp_output.sum().backward()
        python_output.sum().backward()
        for p in cpp_linear.parameters():
            self.assertFalse(p.grad is None)
        self.assertEqual(cpp_parameters['fc.weight'].grad, python_linear.weight.grad)
        self.assertEqual(cpp_parameters['fc.bias'].grad, python_linear.bias.grad)

    def test_cpp_frontend_module_python_inter_op(self):
        if False:
            return 10
        extension = torch.utils.cpp_extension.load(name='cpp_frontend_extension', sources='cpp_extensions/cpp_frontend_extension.cpp', verbose=True)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.x = torch.nn.Parameter(torch.tensor(1.0))
                self.net = extension.Net(3, 5)

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return self.net.forward(input) + self.x
        net = extension.Net(5, 2)
        net.double()
        net.to(torch.get_default_dtype())
        self.assertEqual(str(net), 'Net')
        sequential = torch.nn.Sequential(M(), torch.nn.Tanh(), net, torch.nn.Sigmoid())
        input = torch.randn(2, 3)
        output = sequential.forward(input)
        self.assertEqual(output, sequential(input))
        self.assertEqual(list(output.shape), [2, 2])
        old_dtype = torch.get_default_dtype()
        sequential.to(torch.float64)
        sequential.to(torch.float32)
        sequential.to(old_dtype)
        self.assertEqual(sequential[2].parameters()[0].dtype, old_dtype)
        self.assertEqual(len(list(sequential.parameters())), len(net.parameters()) * 2 + 1)
        self.assertEqual(len(list(sequential.named_parameters())), len(net.named_parameters()) * 2 + 1)
        self.assertEqual(len(list(sequential.buffers())), len(net.buffers()) * 2)
        self.assertEqual(len(list(sequential.modules())), 8)
        net2 = net.clone()
        self.assertEqual(len(net.parameters()), len(net2.parameters()))
        self.assertEqual(len(net.buffers()), len(net2.buffers()))
        self.assertEqual(len(net.modules()), len(net2.modules()))
        for parameter in net.parameters():
            self.assertIsNone(parameter.grad)
        output.sum().backward()
        for parameter in net.parameters():
            self.assertFalse(parameter.grad is None)
            self.assertGreater(parameter.grad.sum(), 0)
        net.zero_grad()
        for p in net.parameters():
            assert p.grad is None, 'zero_grad defaults to setting grads to None'
        self.assertTrue(net.training)
        net.eval()
        self.assertFalse(net.training)
        net.train()
        self.assertTrue(net.training)
        net.eval()
        biased_input = torch.randn(4, 5)
        output_before = net.forward(biased_input)
        bias = net.get_bias().clone()
        self.assertEqual(list(bias.shape), [2])
        net.set_bias(bias + 1)
        self.assertEqual(net.get_bias(), bias + 1)
        output_after = net.forward(biased_input)
        self.assertNotEqual(output_before, output_after)
        self.assertEqual(len(net.parameters()), 2)
        np = net.named_parameters()
        self.assertEqual(len(np), 2)
        self.assertIn('fc.weight', np)
        self.assertIn('fc.bias', np)
        self.assertEqual(len(net.buffers()), 1)
        nb = net.named_buffers()
        self.assertEqual(len(nb), 1)
        self.assertIn('buf', nb)
        self.assertEqual(nb[0][1], torch.eye(5))

    def test_cpp_frontend_module_has_up_to_date_attributes(self):
        if False:
            print('Hello World!')
        extension = torch.utils.cpp_extension.load(name='cpp_frontend_extension', sources='cpp_extensions/cpp_frontend_extension.cpp', verbose=True)
        net = extension.Net(5, 2)
        self.assertEqual(len(net._parameters), 0)
        net.add_new_parameter('foo', torch.eye(5))
        self.assertEqual(len(net._parameters), 1)
        self.assertEqual(len(net._buffers), 1)
        net.add_new_buffer('bar', torch.eye(5))
        self.assertEqual(len(net._buffers), 2)
        self.assertEqual(len(net._modules), 1)
        net.add_new_submodule('fc2')
        self.assertEqual(len(net._modules), 2)

    @unittest.skipIf(not (TEST_CUDA or TEST_ROCM), 'CUDA not found')
    def test_cpp_frontend_module_python_inter_op_with_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        extension = torch.utils.cpp_extension.load(name='cpp_frontend_extension', sources='cpp_extensions/cpp_frontend_extension.cpp', verbose=True)
        net = extension.Net(5, 2)
        for p in net.parameters():
            self.assertTrue(p.device.type == 'cpu')
        cpu_parameters = [p.clone() for p in net.parameters()]
        device = torch.device('cuda', 0)
        net.to(device)
        for (i, p) in enumerate(net.parameters()):
            self.assertTrue(p.device.type == 'cuda')
            self.assertTrue(p.device.index == 0)
            self.assertEqual(cpu_parameters[i], p)
        net.cpu()
        net.add_new_parameter('a', torch.eye(5))
        net.add_new_parameter('b', torch.eye(5))
        net.add_new_buffer('c', torch.eye(5))
        net.add_new_buffer('d', torch.eye(5))
        net.add_new_submodule('fc2')
        net.add_new_submodule('fc3')
        for p in net.parameters():
            self.assertTrue(p.device.type == 'cpu')
        net.cuda()
        for p in net.parameters():
            self.assertTrue(p.device.type == 'cuda')

    def test_returns_shared_library_path_when_is_python_module_is_true(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\n        #include <torch/script.h>\n        torch::Tensor func(torch::Tensor x) { return x; }\n        static torch::RegisterOperators r("test::func", &func);\n        '
        torch.utils.cpp_extension.load_inline(name='is_python_module', cpp_sources=source, functions='func', verbose=True, is_python_module=False)
        self.assertEqual(torch.ops.test.func(torch.eye(5)), torch.eye(5))

    def test_set_default_type_also_changes_aten_default_type(self):
        if False:
            i = 10
            return i + 15
        module = torch.utils.cpp_extension.load_inline(name='test_set_default_type', cpp_sources='torch::Tensor get() { return torch::empty({}); }', functions='get', verbose=True)
        initial_default = torch.get_default_dtype()
        try:
            self.assertEqual(module.get().dtype, initial_default)
            torch.set_default_dtype(torch.float64)
            self.assertEqual(module.get().dtype, torch.float64)
            torch.set_default_dtype(torch.float32)
            self.assertEqual(module.get().dtype, torch.float32)
            torch.set_default_dtype(torch.float16)
            self.assertEqual(module.get().dtype, torch.float16)
        finally:
            torch.set_default_dtype(initial_default)

    def test_compilation_error_formatting(self):
        if False:
            return 10
        with self.assertRaises(RuntimeError) as e:
            torch.utils.cpp_extension.load_inline(name='test_compilation_error_formatting', cpp_sources='int main() { return 0 }')
        pattern = '.*(\\\\n|\\\\r).*'
        self.assertNotRegex(str(e), pattern)

    def test_warning(self):
        if False:
            print('Hello World!')
        source = '\n        // error_type:\n        // 0: no error\n        // 1: torch::TypeError\n        // 2: python_error()\n        // 3: py::error_already_set\n        at::Tensor foo(at::Tensor x, int error_type) {\n            std::ostringstream err_stream;\n            err_stream << "Error with "  << x.type();\n\n            TORCH_WARN(err_stream.str());\n            if(error_type == 1) {\n                throw torch::TypeError(err_stream.str().c_str());\n            }\n            if(error_type == 2) {\n                PyObject* obj = PyTuple_New(-1);\n                TORCH_CHECK(!obj);\n                // Pretend it was caught in a different thread and restored here\n                auto e = python_error();\n                e.persist();\n                e.restore();\n                throw e;\n            }\n            if(error_type == 3) {\n                throw py::key_error(err_stream.str());\n            }\n            return x.cos();\n        }\n        '
        t = torch.rand(2).double()
        cpp_tensor_name = 'CPUDoubleType'
        warn_mod = torch.utils.cpp_extension.load_inline(name='warn_mod', cpp_sources=[source], functions=['foo'], with_pytorch_error_handling=False)
        with warnings.catch_warnings(record=True) as w:
            warn_mod.foo(t, 0)
            self.assertEqual(len(w), 0)
            with self.assertRaisesRegex(TypeError, t.type()):
                warn_mod.foo(t, 1)
            self.assertEqual(len(w), 0)
            with self.assertRaisesRegex(SystemError, 'bad argument to internal function'):
                warn_mod.foo(t, 2)
            self.assertEqual(len(w), 0)
            with self.assertRaisesRegex(KeyError, cpp_tensor_name):
                warn_mod.foo(t, 3)
            self.assertEqual(len(w), 0)
        warn_mod = torch.utils.cpp_extension.load_inline(name='warn_mod', cpp_sources=[source], functions=['foo'], with_pytorch_error_handling=True)
        with warnings.catch_warnings(record=True) as w:
            warn_mod.foo(t, 0)
            self.assertEqual(len(w), 1)
            with self.assertRaisesRegex(TypeError, t.type()):
                warn_mod.foo(t, 1)
            self.assertEqual(len(w), 2)
            with self.assertRaisesRegex(SystemError, 'bad argument to internal function'):
                warn_mod.foo(t, 2)
            self.assertEqual(len(w), 3)
            with self.assertRaisesRegex(KeyError, cpp_tensor_name):
                warn_mod.foo(t, 3)
            self.assertEqual(len(w), 4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('error')
            with self.assertRaisesRegex(UserWarning, t.type()):
                warn_mod.foo(t, 0)
            self.assertEqual(len(w), 0)
            with self.assertRaisesRegex(TypeError, t.type()):
                warn_mod.foo(t, 1)
            self.assertEqual(len(w), 0)

    def test_autograd_from_cpp(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\n        void run_back(at::Tensor x) {\n            x.backward({});\n        }\n\n        void run_back_no_gil(at::Tensor x) {\n            pybind11::gil_scoped_release no_gil;\n            x.backward({});\n        }\n        '

        class MyFn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                return x.clone()

            @staticmethod
            def backward(ctx, gx):
                if False:
                    return 10
                return gx
        test_backward_deadlock = torch.utils.cpp_extension.load_inline(name='test_backward_deadlock', cpp_sources=[source], functions=['run_back', 'run_back_no_gil'])
        inp = torch.rand(20, requires_grad=True)
        loss = MyFn.apply(inp).sum()
        with self.assertRaisesRegex(RuntimeError, 'The autograd engine was called while holding the GIL.'):
            test_backward_deadlock.run_back(loss)
        inp = torch.rand(20, requires_grad=True)
        loss = MyFn.apply(inp).sum()
        test_backward_deadlock.run_back_no_gil(loss)

    def test_custom_compound_op_autograd(self):
        if False:
            print('Hello World!')
        source = '\n        #include <torch/library.h>\n        torch::Tensor my_add(torch::Tensor x, torch::Tensor y) {\n          return x + y;\n        }\n        TORCH_LIBRARY(my, m) {\n            m.def("add", &my_add);\n        }\n        '
        torch.utils.cpp_extension.load_inline(name='is_python_module', cpp_sources=source, verbose=True, is_python_module=False)
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)
        for fast_mode in (True, False):
            gradcheck(torch.ops.my.add, [a, b], eps=0.01, fast_mode=fast_mode)

    def test_custom_functorch_error(self):
        if False:
            print('Hello World!')
        identity_m = torch.utils.cpp_extension.load(name='identity', sources=['cpp_extensions/identity.cpp'])
        t = torch.randn(3, requires_grad=True)
        msg = 'cannot use C\\+\\+ torch::autograd::Function with functorch'
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.func.vmap(identity_m.identity)(t)
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.func.grad(identity_m.identity)(t)

    def test_gen_extension_h_pch(self):
        if False:
            i = 10
            return i + 15
        if not IS_LINUX:
            return
        source = '\n        at::Tensor sin_add(at::Tensor x, at::Tensor y) {\n            return x.sin() + y.sin();\n        }\n        '
        head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
        head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')
        remove_extension_h_precompiler_headers()
        pch_exist = os.path.exists(head_file_pch)
        signature_exist = os.path.exists(head_file_signature)
        self.assertEqual(pch_exist, False)
        self.assertEqual(signature_exist, False)
        torch.utils.cpp_extension.load_inline(name='inline_extension_with_pch', cpp_sources=[source], functions=['sin_add'], verbose=True, use_pch=True)
        pch_exist = os.path.exists(head_file_pch)
        signature_exist = os.path.exists(head_file_signature)
        compiler = get_cxx_compiler()
        if check_compiler_is_gcc(compiler):
            self.assertEqual(pch_exist, True)
            self.assertEqual(signature_exist, True)
if __name__ == '__main__':
    common.run_tests()