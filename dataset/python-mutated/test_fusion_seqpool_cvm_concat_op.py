import sys
import unittest
import numpy as np
from op_test import OpTest
sys.path.append('../../test/sequence')
from test_cvm_op import cvm_compute
from test_sequence_pool import compute_seqpool_avg, compute_seqpool_sqrt, compute_seqpool_sum

def convert_to_offset(lod):
    if False:
        print('Hello World!')
    offset = [[0] for i in lod]
    for (i, level) in enumerate(lod):
        for seq_len in level:
            offset[i].append(offset[i][-1] + seq_len)
    return offset

class TestFusionSeqPoolCVMConcatOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.w = 11
        self.use_cvm = True
        self.lods = [[[2, 3, 5]], [[1, 5, 2]]]
        self.set_conf()
        self.set_pooltype()
        self.op_type = 'fusion_seqpool_cvm_concat'
        self.axis = 1
        bs = len(self.lods[0][0])
        inputs = []
        outs = []
        cvm = np.array([[0.6, 0.4]]).astype('float32')
        i = 0
        for lod in self.lods:
            assert bs == len(lod[0]), 'All lod size should be equal'
            x = np.random.uniform(0.1, 1, [sum(lod[0]), self.w]).astype('float32')
            offset = convert_to_offset(lod)
            out = np.zeros((bs, self.w)).astype('float32')
            if self.pooltype == 'SUM':
                compute_seqpool_sum(x, offset, out)
                out = cvm_compute(out, self.w, self.use_cvm)
            elif self.pooltype == 'AVERAGE':
                compute_seqpool_avg(x, offset, out)
                out = cvm_compute(out, self.w, self.use_cvm)
            elif self.pooltype == 'SQRT':
                compute_seqpool_sqrt(x, offset, out)
                out = cvm_compute(out, self.w, self.use_cvm)
            else:
                raise Exception('Unsupported pool type!')
            inputs.append((f'x_{i}', (x, lod)))
            outs.append(out)
            i = i + 1
        self.inputs = {'X': inputs, 'CVM': cvm}
        self.outputs = {'Out': np.concatenate(outs, axis=self.axis)}
        self.attrs = {'pooltype': self.pooltype, 'axis': self.axis}

    def set_pooltype(self):
        if False:
            for i in range(10):
                print('nop')
        self.pooltype = 'SUM'

    def set_conf(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

class TestFusionSeqPoolCVMConcatOpCase1(TestFusionSeqPoolCVMConcatOp):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.lods = [[[1]]]

class TestFusionSeqPoolCVMConcatOpCase2(TestFusionSeqPoolCVMConcatOp):

    def set_conf(self):
        if False:
            i = 10
            return i + 15
        self.lods = [[[1]], [[1]], [[1]]]

class TestFusionSeqPoolCVMConcatOpCase3(TestFusionSeqPoolCVMConcatOp):

    def set_conf(self):
        if False:
            print('Hello World!')
        self.lods = [[[1, 3, 4, 6]]]
        self.w = 10

class TestFusionSeqPoolCVMConcatOpCase4(TestFusionSeqPoolCVMConcatOp):

    def set_conf(self):
        if False:
            return 10
        self.lods = [[[2, 13, 4]], [[1, 1, 1]], [[5, 3, 1]], [[9, 10, 3]]]
        self.w = 3

def create_test_avg_sqrt_class(parent):
    if False:
        while True:
            i = 10

    class TestSeqPoolAvgCase(parent):

        def set_pooltype(self):
            if False:
                i = 10
                return i + 15
            self.pooltype = 'AVERAGE'

    class TestSeqPoolSqrtCase(parent):

        def set_pooltype(self):
            if False:
                i = 10
                return i + 15
            self.pooltype = 'SQRT'
    cls_name_avg = '{}_{}'.format(parent.__name__, 'avg')
    cls_name_sqrt = '{}_{}'.format(parent.__name__, 'sqrt')
    TestSeqPoolAvgCase.__name__ = cls_name_avg
    TestSeqPoolSqrtCase.__name__ = cls_name_sqrt
    globals()[cls_name_avg] = TestSeqPoolAvgCase
    globals()[cls_name_sqrt] = TestSeqPoolSqrtCase
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOp)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase1)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase2)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase3)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase4)
if __name__ == '__main__':
    unittest.main()