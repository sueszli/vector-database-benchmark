import os
import sys
import torch
import unittest
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestUnsupportedOps(JitTestCase):

    def test_factory_ops_requires_grad_fail(self):
        if False:
            while True:
                i = 10

        def ones():
            if False:
                while True:
                    i = 10
            return torch.ones([2], requires_grad=True)
        with self.assertRaisesRegexWithHighlight(Exception, 'Keyword argument requires_grad unknown', 'torch.ones'):
            torch.jit.script(ones)

        def randn():
            if False:
                while True:
                    i = 10
            return torch.randn([2], requires_grad=True)
        with self.assertRaisesRegexWithHighlight(Exception, 'Keyword argument requires_grad unknown', 'torch.randn'):
            torch.jit.script(randn)

        def zeros():
            if False:
                i = 10
                return i + 15
            return torch.zeros([2], requires_grad=True)
        with self.assertRaisesRegexWithHighlight(Exception, 'Keyword argument requires_grad unknown', 'torch.zeros'):
            torch.jit.script(zeros)

    @unittest.skipIf(not torch._C.has_lapack, 'PyTorch compiled without Lapack')
    def test_init_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def calculate_gain():
            if False:
                return 10
            return torch.nn.init.calculate_gain('leaky_relu', 0.2)

        def eye_():
            if False:
                i = 10
                return i + 15
            return torch.nn.init.eye_(torch.zeros([2, 2]))

        def dirac_():
            if False:
                while True:
                    i = 10
            return torch.nn.init.dirac_(torch.empty(3, 16, 5, 5))

        def kaiming_uniform_():
            if False:
                while True:
                    i = 10
            return torch.nn.init.kaiming_normal_(torch.empty(3, 5))

        def orthogonal_():
            if False:
                print('Hello World!')
            return torch.nn.init.orthogonal_(torch.empty(3, 5))

        def sparse():
            if False:
                for i in range(10):
                    print('nop')
            return torch.nn.init.sparse_(torch.empty(3, 5), sparsity=0.1)
        for func in [calculate_gain, eye_, dirac_, kaiming_uniform_, orthogonal_, sparse]:
            func()
            with self.assertRaisesRegex(Exception, ''):
                torch.jit.script(func)