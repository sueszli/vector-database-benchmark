import unittest
from torch._lazy.ts_backend import init as init_ts_backend
init_ts_backend()
from torch._lazy import config
from torch._lazy.extract_compiled_graph import extract_compiled_graph
import torch
from torch import nn
import dis
import inspect
from torch import fx
import re
from contextlib import contextmanager
import copy

class ModuleConstScale(nn.Module):

    def forward(self, a):
        if False:
            return 10
        return a * 2

class ModuleSub(nn.Module):

    def forward(self, a, b):
        if False:
            print('Hello World!')
        return a - b

class ModuleAddcmul(nn.Module):
    """
    addcmul function takes a at::Scalar which results in a special TSData containing a Scalar rather than a Tensor.
    """

    def forward(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        return torch.addcmul(a, b, c, value=5)

class ModuleReturnMulti(nn.Module):

    def forward(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return (b + 1, a - 1)

class ModuleReturnDupTensor(nn.Module):
    """
    Handle the corner case that the same tensor appears multiple times in the
    returned tuple. torchbench like drq will hit this corner case when running
    thru torchdynamo..
    """

    def forward(self, a, b):
        if False:
            print('Hello World!')
        c = a + b
        return (a - b, c, a + 1, c)

class ModuleInplaceUpdate(nn.Module):

    def forward(self, a, b):
        if False:
            i = 10
            return i + 15
        a.sub_(b)
        return (b - 1, b + 1)

@contextmanager
def force_fallback_ctx_mgr(fallback_op):
    if False:
        i = 10
        return i + 15
    oldconfig = config.get_force_fallback()
    config.set_force_fallback(fallback_op)
    try:
        yield None
    finally:
        config.set_force_fallback(oldconfig)

@contextmanager
def nop_ctx_mgr():
    if False:
        for i in range(10):
            print('nop')
    try:
        yield None
    finally:
        pass

def gen_rand_args(mod):
    if False:
        i = 10
        return i + 15
    args = []
    for _ in range(len(inspect.signature(mod.forward).parameters)):
        args.append(torch.randn(2, 3))
    return args

def allclose(expected, actual):
    if False:
        for i in range(10):
            print('nop')

    def unwrap(cont):
        if False:
            print('Hello World!')
        if isinstance(cont, (list, tuple)) and len(cont) == 1:
            return cont[0]
        return cont
    expected = unwrap(expected)
    actual = unwrap(actual)
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.allclose(expected, actual)
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all((torch.allclose(a, b) for (a, b) in zip(expected, actual)))
    else:
        raise RuntimeError('Unexpected types')

def verify_reusing_compiled_graph(mod, exception_msg_pattern, ncase=10):
    if False:
        while True:
            i = 10
    args = gen_rand_args(mod)
    out = mod(*args)
    dis.dis(mod.forward)
    try:
        optimized_mod = extract_compiled_graph(fx.symbolic_trace(mod), args)
    except RuntimeError as e:
        if exception_msg_pattern is None:
            raise e
        exception_message = str(e)
        if not re.search(exception_msg_pattern, exception_message):
            raise RuntimeError(f'Exception message does not match the required pattern: {exception_message}') from e
        else:
            return
    if exception_msg_pattern is not None:
        raise RuntimeError(f'Expect an exception matching pattern {exception_msg_pattern}')
    print('return value of optimized_mod', optimized_mod(*args))
    failed_index = []
    for i in range(ncase):
        rand_args = gen_rand_args(mod)
        rand_args_copy = copy.deepcopy(rand_args)
        expected = mod(*rand_args)
        actual = optimized_mod(*rand_args_copy)
        if not allclose(expected, actual):
            print(f'Incorrect results. expected {expected}, actual {actual}')
            failed_index.append(i)
            continue
        if not allclose(rand_args, rand_args_copy):
            print(f'Incorrect updated arguments. expected {rand_args}, actual {rand_args_copy}')
            failed_index.append(i)
            continue
    if len(failed_index) > 0:
        raise RuntimeError(f'Failed {len(failed_index)}/{ncase} cases')

def maketest(module_cls, exception_msg_pattern=None, ctxmgr=None):
    if False:
        while True:
            i = 10

    def wrapper(self):
        if False:
            while True:
                i = 10
        nonlocal ctxmgr
        if not ctxmgr:
            ctxmgr = nop_ctx_mgr()
        with ctxmgr:
            verify_reusing_compiled_graph(module_cls(), exception_msg_pattern)
    return wrapper

class OptimizeTest(unittest.TestCase):
    test_sub = maketest(ModuleSub)
    test_ltc_fallback = maketest(ModuleSub, exception_msg_pattern='fallback.*aten::sub', ctxmgr=force_fallback_ctx_mgr('aten::sub'))
    test_const_scale = maketest(ModuleConstScale)
    test_addcmul = maketest(ModuleAddcmul)
    test_return_multi = maketest(ModuleReturnMulti)
    test_return_dup_tensor = maketest(ModuleReturnDupTensor)
    test_inplace_update = maketest(ModuleInplaceUpdate)