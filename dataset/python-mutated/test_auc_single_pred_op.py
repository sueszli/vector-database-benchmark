import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestAucSinglePredOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'auc'
        pred = np.random.random((128, 2)).astype('float32')
        pred0 = pred[:, 0].reshape(128, 1)
        labels = np.random.randint(0, 2, (128, 1)).astype('int64')
        num_thresholds = 200
        slide_steps = 1
        stat_pos = np.zeros((1 + slide_steps) * (num_thresholds + 1) + 1).astype('int64')
        stat_neg = np.zeros((1 + slide_steps) * (num_thresholds + 1) + 1).astype('int64')
        self.inputs = {'Predict': pred0, 'Label': labels, 'StatPos': stat_pos, 'StatNeg': stat_neg}
        self.attrs = {'curve': 'ROC', 'num_thresholds': num_thresholds, 'slide_steps': slide_steps}
        python_auc = paddle.metric.Auc(name='auc', curve='ROC', num_thresholds=num_thresholds)
        for i in range(128):
            pred[i][1] = pred[i][0]
        python_auc.update(pred, labels)
        pos = python_auc._stat_pos.tolist() * 2
        pos.append(1)
        neg = python_auc._stat_neg.tolist() * 2
        neg.append(1)
        self.outputs = {'AUC': np.array(python_auc.accumulate()), 'StatPosOut': np.array(pos), 'StatNegOut': np.array(neg)}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

class TestAucGlobalSinglePredOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'auc'
        pred = np.random.random((128, 2)).astype('float32')
        pred0 = pred[:, 0].reshape(128, 1)
        labels = np.random.randint(0, 2, (128, 1)).astype('int64')
        num_thresholds = 200
        slide_steps = 0
        stat_pos = np.zeros((1, num_thresholds + 1)).astype('int64')
        stat_neg = np.zeros((1, num_thresholds + 1)).astype('int64')
        self.inputs = {'Predict': pred0, 'Label': labels, 'StatPos': stat_pos, 'StatNeg': stat_neg}
        self.attrs = {'curve': 'ROC', 'num_thresholds': num_thresholds, 'slide_steps': slide_steps}
        python_auc = paddle.metric.Auc(name='auc', curve='ROC', num_thresholds=num_thresholds)
        for i in range(128):
            pred[i][1] = pred[i][0]
        python_auc.update(pred, labels)
        pos = python_auc._stat_pos
        neg = python_auc._stat_neg
        self.outputs = {'AUC': np.array(python_auc.accumulate()), 'StatPosOut': np.array([pos]), 'StatNegOut': np.array([neg])}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_dygraph=False)
if __name__ == '__main__':
    unittest.main()