import torch
import re
import unittest
from subprocess import CalledProcessError
from torch._inductor.codecache import CppCodeCache
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import LazyVal, IS_FBCODE
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase

def test_cpu():
    if False:
        print('Hello World!')
    try:
        CppCodeCache.load('')
        return not IS_FBCODE
    except (CalledProcessError, OSError, torch._inductor.exc.InvalidCxxCompiler, torch._inductor.exc.CppCompileError):
        return False
HAS_CPU = LazyVal(test_cpu)
HAS_CUDA = has_triton()

@register_backend
def count_bytes_inductor(gm, example_inputs):
    if False:
        return 10
    return compile_fx(gm, example_inputs, inner_compile=count_bytes_inner)

def _check_has_dynamic_shape(self: TestCase, code):
    if False:
        return 10
    for_loop_found = False
    has_dynamic = False
    lines = code.split('\n')
    for line in lines:
        if 'for(' in line:
            for_loop_found = True
            if re.search(';.*ks.*;', line) is not None:
                has_dynamic = True
                break
    self.assertTrue(has_dynamic, msg=f'Failed to find dynamic for loop variable\n{code}')
    self.assertTrue(for_loop_found, f'Failed to find for loop\n{code}')

def skipCUDAIf(cond, msg):
    if False:
        while True:
            i = 10
    if cond:

        def decorate_fn(fn):
            if False:
                i = 10
                return i + 15

            def inner(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                if self.device == 'cuda':
                    raise unittest.SkipTest(msg)
                return fn(self, *args, **kwargs)
            return inner
    else:

        def decorate_fn(fn):
            if False:
                for i in range(10):
                    print('nop')
            return fn
    return decorate_fn