import os
import sys
import torch
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestTensorCreationOps(JitTestCase):
    """
    A suite of tests for ops that create tensors.
    """

    def test_randperm_default_dtype(self):
        if False:
            return 10

        def randperm(x: int):
            if False:
                for i in range(10):
                    print('nop')
            perm = torch.randperm(x)
            assert perm.dtype == torch.int64
        self.checkScript(randperm, (3,))

    def test_randperm_specifed_dtype(self):
        if False:
            return 10

        def randperm(x: int):
            if False:
                while True:
                    i = 10
            perm = torch.randperm(x, dtype=torch.float)
            assert perm.dtype == torch.float
        self.checkScript(randperm, (3,))

    def test_triu_indices_default_dtype(self):
        if False:
            i = 10
            return i + 15

        def triu_indices(rows: int, cols: int):
            if False:
                return 10
            indices = torch.triu_indices(rows, cols)
            assert indices.dtype == torch.int64
        self.checkScript(triu_indices, (3, 3))

    def test_triu_indices_specified_dtype(self):
        if False:
            print('Hello World!')

        def triu_indices(rows: int, cols: int):
            if False:
                print('Hello World!')
            indices = torch.triu_indices(rows, cols, dtype=torch.int32)
            assert indices.dtype == torch.int32
        self.checkScript(triu_indices, (3, 3))

    def test_tril_indices_default_dtype(self):
        if False:
            print('Hello World!')

        def tril_indices(rows: int, cols: int):
            if False:
                print('Hello World!')
            indices = torch.tril_indices(rows, cols)
            assert indices.dtype == torch.int64
        self.checkScript(tril_indices, (3, 3))

    def test_tril_indices_specified_dtype(self):
        if False:
            return 10

        def tril_indices(rows: int, cols: int):
            if False:
                while True:
                    i = 10
            indices = torch.tril_indices(rows, cols, dtype=torch.int32)
            assert indices.dtype == torch.int32
        self.checkScript(tril_indices, (3, 3))