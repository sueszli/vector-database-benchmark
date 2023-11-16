import os
import sys
import unittest
import torch
import torch._C
from pathlib import Path
from torch.testing._internal.common_utils import IS_FBCODE
if not IS_FBCODE:
    from test_nnapi import TestNNAPI
    HAS_TEST_NNAPI = True
else:
    from torch.testing._internal.common_utils import TestCase as TestNNAPI
    HAS_TEST_NNAPI = False
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')
'\nUnit Tests for Nnapi backend with delegate\nInherits most tests from TestNNAPI, which loads Android NNAPI models\nwithout the delegate API.\n'
torch_root = Path(__file__).resolve().parent.parent.parent
lib_path = torch_root / 'build' / 'lib' / 'libnnapi_backend.so'

@unittest.skipIf(not os.path.exists(lib_path), 'Skipping the test as libnnapi_backend.so was not found')
@unittest.skipIf(IS_FBCODE, 'test_nnapi.py not found')
class TestNnapiBackend(TestNNAPI):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        module = torch.nn.PReLU()
        self.default_dtype = module.weight.dtype
        torch.set_default_dtype(torch.float32)
        torch.ops.load_library(str(lib_path))

    def call_lowering_to_nnapi(self, traced_module, args):
        if False:
            while True:
                i = 10
        compile_spec = {'forward': {'inputs': args}}
        return torch._C._jit_to_backend('nnapi', traced_module, compile_spec)

    def test_tensor_input(self):
        if False:
            for i in range(10):
                print('nop')
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        module = torch.nn.PReLU()
        traced = torch.jit.trace(module, args)
        self.call_lowering_to_nnapi(traced, args)
        self.call_lowering_to_nnapi(traced, [args])

    def test_compile_spec_santiy(self):
        if False:
            print('Hello World!')
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        module = torch.nn.PReLU()
        traced = torch.jit.trace(module, args)
        errorMsgTail = '\nmethod_compile_spec should contain a Tensor or Tensor List which bundles input parameters: shape, dtype, quantization, and dimorder.\nFor input shapes, use 0 for run/load time flexible input.\nmethod_compile_spec must use the following format:\n{"forward": {"inputs": at::Tensor}} OR {"forward": {"inputs": c10::List<at::Tensor>}}'
        compile_spec = {'backward': {'inputs': args}}
        with self.assertRaisesRegex(RuntimeError, 'method_compile_spec does not contain the "forward" key.' + errorMsgTail):
            torch._C._jit_to_backend('nnapi', traced, compile_spec)
        compile_spec = {'forward': 1}
        with self.assertRaisesRegex(RuntimeError, 'method_compile_spec does not contain a dictionary with an "inputs" key, under it\'s "forward" key.' + errorMsgTail):
            torch._C._jit_to_backend('nnapi', traced, compile_spec)
        compile_spec = {'forward': {'not inputs': args}}
        with self.assertRaisesRegex(RuntimeError, 'method_compile_spec does not contain a dictionary with an "inputs" key, under it\'s "forward" key.' + errorMsgTail):
            torch._C._jit_to_backend('nnapi', traced, compile_spec)
        compile_spec = {'forward': {'inputs': 1}}
        with self.assertRaisesRegex(RuntimeError, 'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.' + errorMsgTail):
            torch._C._jit_to_backend('nnapi', traced, compile_spec)
        compile_spec = {'forward': {'inputs': [1]}}
        with self.assertRaisesRegex(RuntimeError, 'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.' + errorMsgTail):
            torch._C._jit_to_backend('nnapi', traced, compile_spec)

    def tearDown(self):
        if False:
            print('Hello World!')
        torch.set_default_dtype(self.default_dtype)