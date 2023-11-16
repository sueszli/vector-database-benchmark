import unittest
import numpy as np
from op_test import OpTest, randomize_probability
import paddle

class TestBprLossOp1(OpTest):
    """Test BprLoss with discrete one-hot labels."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'bpr_loss'
        batch_size = 40
        class_num = 5
        X = randomize_probability(batch_size, class_num, dtype='float64')
        label = np.random.randint(0, class_num, (batch_size, 1), dtype='int64')
        bpr_loss_result = []
        for i in range(batch_size):
            sum = 0.0
            for j in range(class_num):
                if j == label[i][0]:
                    continue
                sum += -np.log(1.0 + np.exp(X[i][j] - X[i][label[i][0]]))
            bpr_loss_result.append(-sum / (class_num - 1))
        bpr_loss = np.asmatrix([[x] for x in bpr_loss_result], dtype='float64')
        self.inputs = {'X': X, 'Label': label}
        self.outputs = {'Y': bpr_loss}

    def test_check_output(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.check_output(check_dygraph=False)
        paddle.disable_static()

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Y', numeric_grad_delta=0.001, check_dygraph=False)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()