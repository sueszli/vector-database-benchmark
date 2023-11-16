import torch
from torch.cuda.jiterator import _create_jit_fn as create_jit_fn
from torch.cuda.jiterator import _create_multi_output_jit_fn as create_multi_output_jit_fn
import sys
from itertools import product
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, TEST_CUDA, NoTest
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_device_type import skipCUDAIfVersionLessThan, instantiate_device_type_tests, dtypes, toleranceOverride, tol
if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = NoTest
code_string = 'template <typename T> T my_fused_kernel(T x, T y, T alpha, T beta) { return alpha * x + beta * y; }'
jitted_fn = create_jit_fn(code_string, alpha=1, beta=1)

def ref_fn(x, y, alpha=1, beta=1):
    if False:
        print('Hello World!')
    return alpha * x + beta * y

class TestPythonJiterator(TestCase):

    @parametrize('shape_strides', [(([3, 3], [3, 1]), ([3, 3], [3, 1]))])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16), all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_contiguous(self, device, dtypes, shape_strides):
        if False:
            i = 10
            return i + 15
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])
        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])
        expected = ref_fn(a, b)
        result = jitted_fn(a, b)
        self.assertEqual(expected, result)

    @skipCUDAIfVersionLessThan((11, 6))
    @parametrize('shape_strides', [(([3, 3], [1, 3]), ([3, 1], [1, 3]))])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16), all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_noncontiguous(self, device, dtypes, shape_strides):
        if False:
            for i in range(10):
                print('nop')
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])
        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])
        expected = ref_fn(a, b)
        result = jitted_fn(a, b)
        self.assertEqual(expected, result)

    @dtypes(torch.float, torch.double, torch.float16, torch.bfloat16)
    @parametrize('alpha', [-1, 2.0, None])
    @parametrize('beta', [3, -4.2, None])
    @toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.001)})
    def test_extra_args(self, device, dtype, alpha, beta):
        if False:
            while True:
                i = 10
        a = torch.rand(3, device=device).mul(10).type(dtype)
        b = torch.rand(3, device=device).mul(10).type(dtype)
        extra_args = {}
        if alpha is not None:
            extra_args['alpha'] = alpha
        if beta is not None:
            extra_args['beta'] = beta
        expected = ref_fn(a, b, **extra_args)
        result = jitted_fn(a, b, **extra_args)
        self.assertEqual(expected, result)

    @parametrize('is_train', [True, False])
    def test_bool_extra_args(self, device, is_train):
        if False:
            i = 10
            return i + 15
        code_string = 'template <typename T> T conditional(T x, T mask, bool is_train) { return is_train ? x * mask : x; }'
        jitted_fn = create_jit_fn(code_string, is_train=False)

        def ref_fn(x, mask, is_train):
            if False:
                print('Hello World!')
            return x * mask if is_train else x
        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)
        expected = ref_fn(a, b, is_train=is_train)
        result = jitted_fn(a, b, is_train=is_train)
        self.assertEqual(expected, result)

    def test_multiple_functors(self, device):
        if False:
            for i in range(10):
                print('nop')
        code_string = '\n        template <typename T> T fn(T x, T mask) { return x * mask; }\n        template <typename T> T main_fn(T x, T mask, T y) { return fn(x, mask) + y; }\n        '
        jitted_fn = create_jit_fn(code_string)

        def ref_fn(x, mask, y):
            if False:
                print('Hello World!')
            return x * mask + y
        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)
        c = torch.rand(3, device=device)
        expected = ref_fn(a, b, c)
        result = jitted_fn(a, b, c)
        self.assertEqual(expected, result)

    @parametrize('num_inputs', [1, 5, 8])
    def test_various_num_inputs(self, num_inputs):
        if False:
            print('Hello World!')
        inputs = []
        for i in range(num_inputs):
            inputs.append(torch.rand(3, device='cuda').mul(10))
        input_string = ','.join([f'T i{i}' for i in range(num_inputs)])
        function_body = '+'.join([f'i{i}' for i in range(num_inputs)])
        code_string = f'template <typename T> T my_kernel({input_string}) {{ return {function_body}; }}'
        jitted_fn = create_jit_fn(code_string)

        def ref_fn(*inputs):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sum(torch.stack(inputs), dim=0)
        expected = ref_fn(*inputs)
        result = jitted_fn(*inputs)
        self.assertEqual(expected, result)

    @parametrize('num_outputs', [1, 4, 8])
    def test_various_num_outputs(self, num_outputs):
        if False:
            while True:
                i = 10
        input = torch.rand(3, device='cuda')
        output_string = ', '.join([f'T& out{i}' for i in range(num_outputs)])
        function_body = ''
        for i in range(num_outputs):
            function_body += f'out{i} = input + {i};\n'
        code_string = f'template <typename T> void my_kernel(T input, {output_string}) {{ {function_body} }}'
        jitted_fn = create_multi_output_jit_fn(code_string, num_outputs)

        def ref_fn(input):
            if False:
                print('Hello World!')
            outputs = []
            for i in range(num_outputs):
                outputs.append(input + i)
            if num_outputs == 1:
                return outputs[0]
            return tuple(outputs)
        expected = ref_fn(input)
        result = jitted_fn(input)
        for i in range(num_outputs):
            self.assertEqual(expected[i], result[i])

    @parametrize('code_string', ['template <typename T> T my _kernel(T x) { return x; }', 'template <typename T> Tmy_kernel(T x) { return x; }'])
    def test_invalid_function_name(self, code_string):
        if False:
            print('Hello World!')
        with self.assertRaises(Exception):
            jitted_fn = create_jit_fn(code_string)
instantiate_device_type_tests(TestPythonJiterator, globals(), only_for='cuda')
if __name__ == '__main__':
    run_tests()