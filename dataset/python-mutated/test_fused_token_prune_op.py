import unittest
import numpy as np
from op_test import OpTest
from paddle.framework import core

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFusedTokenPruneOp(OpTest):

    def setDtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32

    def setInouts(self):
        if False:
            return 10
        attn = [[1, 2], [3, 4]]
        attn = np.array(attn, dtype=self.dtype)
        attn = np.expand_dims(attn, axis=0)
        self.attn = np.expand_dims(attn, axis=0)
        mask = [[1, 1], [-1, -1]]
        mask = np.array(mask, dtype=self.dtype)
        mask = np.expand_dims(mask, axis=0)
        self.mask = np.expand_dims(mask, axis=0)
        x = [[1, 2, 3], [4, 5, 6]]
        x = np.array(x, dtype=self.dtype)
        self.x = np.expand_dims(x, axis=0)
        new_mask = [[1]]
        new_mask = np.array(new_mask, dtype=self.dtype)
        new_mask = np.expand_dims(new_mask, axis=0)
        self.new_mask = np.expand_dims(new_mask, axis=0)
        out_slimmedx_py = [[[1, 2, 3]]]
        self.out_slimmedx_py = np.array(out_slimmedx_py, dtype=self.dtype)
        out_cls_inds_py = [[0]]
        self.out_cls_inds_py = np.array(out_cls_inds_py, dtype='int64')

    def setUp(self):
        if False:
            return 10
        self.op_type = 'fused_token_prune'
        self.setDtype()
        self.setInouts()
        self.inputs = {'Attn': self.attn, 'Mask': self.mask, 'X': self.x, 'NewMask': self.new_mask}
        self.outputs = {'SlimmedX': self.out_slimmedx_py, 'CLSInds': self.out_cls_inds_py}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CUDAPlace(0))

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFusedTokenPruneOpFloat64(TestFusedTokenPruneOp):

    def setDtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float64

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestFusedTokenPruneOp2(TestFusedTokenPruneOp):

    def setInouts(self):
        if False:
            print('Hello World!')
        attn = [[[[1, 2, 3, 4], [4, 3, 2, 1], [5, 9, 5, 4], [9, 6, 5, 4]], [[8, 5, 2, 0], [1, 0, 2, 3], [2, 2, 3, 2], [7, 4, 1, 8]]]]
        self.attn = np.array(attn, dtype=self.dtype)
        mask = [[[[-1, -1, -1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]], [[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]]]]
        self.mask = np.array(mask, dtype=self.dtype)
        x = [[[1.1, 1.1, 1.1], [2.2, 2.2, 2.2], [3.3, 3.3, 3.3], [4.4, 4.4, 4.4]]]
        self.x = np.array(x, dtype=self.dtype)
        self.new_mask = np.random.rand(1, 2, 2, 2).astype(self.dtype)
        out_slimmedx_py = [[[1.1, 1.1, 1.1], [4.4, 4.4, 4.4]]]
        self.out_slimmedx_py = np.array(out_slimmedx_py, dtype=self.dtype)
        out_cls_inds_py = [[0, 3]]
        self.out_cls_inds_py = np.array(out_cls_inds_py, dtype='int64')
if __name__ == '__main__':
    unittest.main()