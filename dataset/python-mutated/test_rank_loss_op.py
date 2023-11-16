import unittest
import numpy as np
from op_test import OpTest

class TestRankLossOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'rank_loss'
        shape = (100, 1)
        (label_shape, left_shape, right_shape) = self.set_shape()
        label = np.random.randint(0, 2, size=shape).astype('float32')
        left = np.random.random(shape).astype('float32')
        right = np.random.random(shape).astype('float32')
        loss = np.log(1.0 + np.exp(left - right)) - label * (left - right)
        loss = np.reshape(loss, label_shape)
        self.inputs = {'Label': label.reshape(label_shape), 'Left': left.reshape(left_shape), 'Right': right.reshape(right_shape)}
        self.outputs = {'Out': loss.reshape(label_shape)}

    def set_shape(self):
        if False:
            print('Hello World!')
        batch_size = 100
        return ((batch_size, 1), (batch_size, 1), (batch_size, 1))

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['Left', 'Right'], 'Out')

    def test_check_grad_ignore_left(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Right'], 'Out', no_grad_set=set('Left'))

    def test_check_grad_ignore_right(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Left'], 'Out', no_grad_set=set('Right'))

class TestRankLossOp1(TestRankLossOp):

    def set_shape(self):
        if False:
            print('Hello World!')
        batch_size = 100
        return (batch_size, (batch_size, 1), (batch_size, 1))

class TestRankLossOp2(TestRankLossOp):

    def set_shape(self):
        if False:
            print('Hello World!')
        batch_size = 100
        return ((batch_size, 1), batch_size, (batch_size, 1))

class TestRankLossOp3(TestRankLossOp):

    def set_shape(self):
        if False:
            for i in range(10):
                print('nop')
        batch_size = 100
        return ((batch_size, 1), (batch_size, 1), batch_size)

class TestRankLossOp4(TestRankLossOp):

    def set_shape(self):
        if False:
            return 10
        batch_size = 100
        return (batch_size, batch_size, (batch_size, 1))

class TestRankLossOp5(TestRankLossOp):

    def set_shape(self):
        if False:
            return 10
        batch_size = 100
        return (batch_size, batch_size, batch_size)
if __name__ == '__main__':
    unittest.main()