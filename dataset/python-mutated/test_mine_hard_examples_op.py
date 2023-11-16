import unittest
import numpy as np
from op_test import OpTest

class TestMineHardExamplesOp(OpTest):

    def set_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_test_data()
        self.inputs = {'ClsLoss': self.cls_loss, 'LocLoss': self.loc_loss, 'MatchIndices': self.match_indices, 'MatchDist': self.match_dis}
        self.attrs = {'neg_pos_ratio': self.neg_pos_ratio, 'neg_overlap': self.neg_overlap, 'sample_size': self.sample_size, 'mining_type': self.mining_type}
        self.outputs = {'NegIndices': (self.neg_indices, self.neg_indices_lod), 'UpdatedMatchIndices': self.updated_match_indices}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        return

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'mine_hard_examples'
        self.set_data()

    def init_test_data(self):
        if False:
            return 10
        self.neg_pos_ratio = 1.0
        self.neg_overlap = 0.5
        self.sample_size = 0
        self.mining_type = 'max_negative'
        self.cls_loss = np.array([[0.1, 0.1, 0.3], [0.3, 0.1, 0.1]]).astype('float64')
        self.loc_loss = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.1]]).astype('float64')
        self.match_dis = np.array([[0.2, 0.4, 0.8], [0.1, 0.9, 0.3]]).astype('float64')
        self.match_indices = np.array([[0, -1, -1], [-1, 0, -1]]).astype('int32')
        self.updated_match_indices = self.match_indices
        self.neg_indices_lod = [[1, 1]]
        self.neg_indices = np.array([[1], [0]]).astype('int32')

class TestMineHardExamplesOpHardExample(TestMineHardExamplesOp):

    def init_test_data(self):
        if False:
            print('Hello World!')
        super().init_test_data()
        self.mining_type = 'hard_example'
        self.sample_size = 2
        self.cls_loss = np.array([[0.5, 0.1, 0.3], [0.3, 0.1, 0.1]]).astype('float64')
        self.loc_loss = np.array([[0.2, 0.2, 0.3], [0.3, 0.1, 0.2]]).astype('float64')
        self.match_indices = np.array([[0, -1, -1], [-1, 0, -1]]).astype('int32')
        self.updated_match_indices = np.array([[0, -1, -1], [-1, -1, -1]]).astype('int32')
        self.neg_indices_lod = [[1, 2]]
        self.neg_indices = np.array([[2], [0], [2]]).astype('int32')
if __name__ == '__main__':
    unittest.main()