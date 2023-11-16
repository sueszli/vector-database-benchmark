import sys
import unittest
import numpy as np
from op_test import OpTest
sys.path.append('../../test/sequence')
from test_sequence_pool import compute_seqpool_avg, compute_seqpool_sqrt, compute_seqpool_sum

def convert_to_offset(lod):
    if False:
        i = 10
        return i + 15
    offset = [[0] for i in lod]
    for (i, level) in enumerate(lod):
        for seq_len in level:
            offset[i].append(offset[i][-1] + seq_len)
    return offset

class TestFusionSeqPoolConcatOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.w = 11
        self.lods = [[[2, 3, 5]], [[1, 5, 2]]]
        self.set_conf()
        self.set_pooltype()
        self.op_type = 'fusion_seqpool_concat'
        self.axis = 1
        bs = len(self.lods[0][0])
        inputs = []
        outs = []
        i = 0
        for lod in self.lods:
            assert bs == len(lod[0]), 'All lod size should be equal'
            x = np.random.uniform(0.1, 1, [sum(lod[0]), self.w]).astype('float32')
            offset = convert_to_offset(lod)
            out = np.zeros((bs, self.w)).astype('float32')
            if self.pooltype == 'SUM':
                compute_seqpool_sum(x, offset, out)
            elif self.pooltype == 'AVERAGE':
                compute_seqpool_avg(x, offset, out)
            elif self.pooltype == 'SQRT':
                compute_seqpool_sqrt(x, offset, out)
            else:
                raise Exception('Unsupported pool type!')
            inputs.append((f'x_{i}', (x, lod)))
            outs.append(out)
            i = i + 1
        self.inputs = {'X': inputs}
        self.outputs = {'Out': np.concatenate(outs, axis=self.axis)}
        self.attrs = {'pooltype': self.pooltype, 'axis': self.axis}

    def set_pooltype(self):
        if False:
            while True:
                i = 10
        self.pooltype = 'SUM'

    def set_conf(self):
        if False:
            return 10
        pass

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

class TestFusionSeqPoolConcatOpCase1(TestFusionSeqPoolConcatOp):

    def set_conf(self):
        if False:
            i = 10
            return i + 15
        self.lods = [[[1]]]

class TestFusionSeqPoolConcatOpCase2(TestFusionSeqPoolConcatOp):

    def set_conf(self):
        if False:
            print('Hello World!')
        self.lods = [[[1]], [[1]], [[1]]]

class TestFusionSeqPoolConcatOpCase3(TestFusionSeqPoolConcatOp):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.lods = [[[1, 3, 4, 6]]]
        self.w = 10

class TestFusionSeqPoolConcatOpCase4(TestFusionSeqPoolConcatOp):

    def set_conf(self):
        if False:
            for i in range(10):
                print('nop')
        self.lods = [[[2, 13, 4]], [[1, 1, 1]], [[5, 3, 1]], [[9, 10, 3]]]
        self.w = 3

def create_test_avg_sqrt_class(parent):
    if False:
        return 10

    class TestSeqPoolAvgCase(parent):

        def set_pooltype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.pooltype = 'AVERAGE'

    class TestSeqPoolSqrtCase(parent):

        def set_pooltype(self):
            if False:
                print('Hello World!')
            self.pooltype = 'SQRT'
    cls_name_avg = '{}_{}'.format(parent.__name__, 'avg')
    cls_name_sqrt = '{}_{}'.format(parent.__name__, 'sqrt')
    TestSeqPoolAvgCase.__name__ = cls_name_avg
    TestSeqPoolSqrtCase.__name__ = cls_name_sqrt
    globals()[cls_name_avg] = TestSeqPoolAvgCase
    globals()[cls_name_sqrt] = TestSeqPoolSqrtCase
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOp)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase1)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase2)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase3)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase4)
if __name__ == '__main__':
    unittest.main()