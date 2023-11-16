from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np

def generic_func_1():
    if False:
        i = 10
        return i + 15
    pass

def cuda_func_1():
    if False:
        i = 10
        return i + 15
    pass

def generic_func_2():
    if False:
        while True:
            i = 10
    pass

def cuda_func_2():
    if False:
        return 10
    pass

def generic_calls_generic():
    if False:
        i = 10
        return i + 15
    pass

def generic_calls_cuda():
    if False:
        while True:
            i = 10
    pass

def cuda_calls_generic():
    if False:
        while True:
            i = 10
    pass

def cuda_calls_cuda():
    if False:
        while True:
            i = 10
    pass

def target_overloaded():
    if False:
        return 10
    pass

def generic_calls_target_overloaded():
    if False:
        return 10
    pass

def cuda_calls_target_overloaded():
    if False:
        for i in range(10):
            print('nop')
    pass

def target_overloaded_calls_target_overloaded():
    if False:
        i = 10
        return i + 15
    pass
GENERIC_FUNCTION_1 = 2
CUDA_FUNCTION_1 = 3
GENERIC_FUNCTION_2 = 5
CUDA_FUNCTION_2 = 7
GENERIC_CALLS_GENERIC = 11
GENERIC_CALLS_CUDA = 13
CUDA_CALLS_GENERIC = 17
CUDA_CALLS_CUDA = 19
GENERIC_TARGET_OL = 23
CUDA_TARGET_OL = 29
GENERIC_CALLS_TARGET_OL = 31
CUDA_CALLS_TARGET_OL = 37
GENERIC_TARGET_OL_CALLS_TARGET_OL = 41
CUDA_TARGET_OL_CALLS_TARGET_OL = 43

@overload(generic_func_1, target='generic')
def ol_generic_func_1(x):
    if False:
        for i in range(10):
            print('nop')

    def impl(x):
        if False:
            while True:
                i = 10
        x[0] *= GENERIC_FUNCTION_1
    return impl

@overload(cuda_func_1, target='cuda')
def ol_cuda_func_1(x):
    if False:
        return 10

    def impl(x):
        if False:
            return 10
        x[0] *= CUDA_FUNCTION_1
    return impl

@overload(generic_func_2, target='generic')
def ol_generic_func_2(x):
    if False:
        while True:
            i = 10

    def impl(x):
        if False:
            return 10
        x[0] *= GENERIC_FUNCTION_2
    return impl

@overload(cuda_func_2, target='cuda')
def ol_cuda_func(x):
    if False:
        return 10

    def impl(x):
        if False:
            while True:
                i = 10
        x[0] *= CUDA_FUNCTION_2
    return impl

@overload(generic_calls_generic, target='generic')
def ol_generic_calls_generic(x):
    if False:
        for i in range(10):
            print('nop')

    def impl(x):
        if False:
            return 10
        x[0] *= GENERIC_CALLS_GENERIC
        generic_func_1(x)
    return impl

@overload(generic_calls_cuda, target='generic')
def ol_generic_calls_cuda(x):
    if False:
        print('Hello World!')

    def impl(x):
        if False:
            for i in range(10):
                print('nop')
        x[0] *= GENERIC_CALLS_CUDA
        cuda_func_1(x)
    return impl

@overload(cuda_calls_generic, target='cuda')
def ol_cuda_calls_generic(x):
    if False:
        while True:
            i = 10

    def impl(x):
        if False:
            print('Hello World!')
        x[0] *= CUDA_CALLS_GENERIC
        generic_func_1(x)
    return impl

@overload(cuda_calls_cuda, target='cuda')
def ol_cuda_calls_cuda(x):
    if False:
        print('Hello World!')

    def impl(x):
        if False:
            return 10
        x[0] *= CUDA_CALLS_CUDA
        cuda_func_1(x)
    return impl

@overload(target_overloaded, target='generic')
def ol_target_overloaded_generic(x):
    if False:
        print('Hello World!')

    def impl(x):
        if False:
            print('Hello World!')
        x[0] *= GENERIC_TARGET_OL
    return impl

@overload(target_overloaded, target='cuda')
def ol_target_overloaded_cuda(x):
    if False:
        i = 10
        return i + 15

    def impl(x):
        if False:
            print('Hello World!')
        x[0] *= CUDA_TARGET_OL
    return impl

@overload(generic_calls_target_overloaded, target='generic')
def ol_generic_calls_target_overloaded(x):
    if False:
        for i in range(10):
            print('nop')

    def impl(x):
        if False:
            i = 10
            return i + 15
        x[0] *= GENERIC_CALLS_TARGET_OL
        target_overloaded(x)
    return impl

@overload(cuda_calls_target_overloaded, target='cuda')
def ol_cuda_calls_target_overloaded(x):
    if False:
        while True:
            i = 10

    def impl(x):
        if False:
            while True:
                i = 10
        x[0] *= CUDA_CALLS_TARGET_OL
        target_overloaded(x)
    return impl

@overload(target_overloaded_calls_target_overloaded, target='generic')
def ol_generic_calls_target_overloaded_generic(x):
    if False:
        i = 10
        return i + 15

    def impl(x):
        if False:
            print('Hello World!')
        x[0] *= GENERIC_TARGET_OL_CALLS_TARGET_OL
        target_overloaded(x)
    return impl

@overload(target_overloaded_calls_target_overloaded, target='cuda')
def ol_generic_calls_target_overloaded_cuda(x):
    if False:
        for i in range(10):
            print('nop')

    def impl(x):
        if False:
            while True:
                i = 10
        x[0] *= CUDA_TARGET_OL_CALLS_TARGET_OL
        target_overloaded(x)
    return impl

@skip_on_cudasim('Overloading not supported in cudasim')
class TestOverload(CUDATestCase):

    def check_overload(self, kernel, expected):
        if False:
            return 10
        x = np.ones(1, dtype=np.int32)
        cuda.jit(kernel)[1, 1](x)
        self.assertEqual(x[0], expected)

    def check_overload_cpu(self, kernel, expected):
        if False:
            print('Hello World!')
        x = np.ones(1, dtype=np.int32)
        njit(kernel)(x)
        self.assertEqual(x[0], expected)

    def test_generic(self):
        if False:
            print('Hello World!')

        def kernel(x):
            if False:
                print('Hello World!')
            generic_func_1(x)
        expected = GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda(self):
        if False:
            print('Hello World!')

        def kernel(x):
            if False:
                return 10
            cuda_func_1(x)
        expected = CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_and_cuda(self):
        if False:
            for i in range(10):
                print('nop')

        def kernel(x):
            if False:
                i = 10
                return i + 15
            generic_func_1(x)
            cuda_func_1(x)
        expected = GENERIC_FUNCTION_1 * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_two_generic_calls(self):
        if False:
            print('Hello World!')

        def kernel(x):
            if False:
                while True:
                    i = 10
            generic_func_1(x)
            generic_func_2(x)
        expected = GENERIC_FUNCTION_1 * GENERIC_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_call_two_cuda_calls(self):
        if False:
            for i in range(10):
                print('nop')

        def kernel(x):
            if False:
                for i in range(10):
                    print('nop')
            cuda_func_1(x)
            cuda_func_2(x)
        expected = CUDA_FUNCTION_1 * CUDA_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_generic_calls_generic(self):
        if False:
            i = 10
            return i + 15

        def kernel(x):
            if False:
                return 10
            generic_calls_generic(x)
        expected = GENERIC_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_calls_cuda(self):
        if False:
            return 10

        def kernel(x):
            if False:
                while True:
                    i = 10
            generic_calls_cuda(x)
        expected = GENERIC_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_generic(self):
        if False:
            i = 10
            return i + 15

        def kernel(x):
            if False:
                print('Hello World!')
            cuda_calls_generic(x)
        expected = CUDA_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_cuda(self):
        if False:
            i = 10
            return i + 15

        def kernel(x):
            if False:
                while True:
                    i = 10
            cuda_calls_cuda(x)
        expected = CUDA_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_target_overloaded(self):
        if False:
            i = 10
            return i + 15

        def kernel(x):
            if False:
                for i in range(10):
                    print('nop')
            target_overloaded(x)
        expected = CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_generic_calls_target_overloaded(self):
        if False:
            while True:
                i = 10

        def kernel(x):
            if False:
                print('Hello World!')
            generic_calls_target_overloaded(x)
        expected = GENERIC_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_cuda_calls_target_overloaded(self):
        if False:
            while True:
                i = 10

        def kernel(x):
            if False:
                for i in range(10):
                    print('nop')
            cuda_calls_target_overloaded(x)
        expected = CUDA_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_target_overloaded_calls_target_overloaded(self):
        if False:
            while True:
                i = 10

        def kernel(x):
            if False:
                print('Hello World!')
            target_overloaded_calls_target_overloaded(x)
        expected = CUDA_TARGET_OL_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)
        expected = GENERIC_TARGET_OL_CALLS_TARGET_OL * GENERIC_TARGET_OL
        self.check_overload_cpu(kernel, expected)
if __name__ == '__main__':
    unittest.main()