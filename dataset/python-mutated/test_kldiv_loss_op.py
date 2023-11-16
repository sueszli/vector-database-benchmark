import unittest
import numpy as np
from op_test import OpTest, paddle_static_guard
import paddle
from paddle.nn.functional import kl_div

def kldiv_loss(x, target, reduction):
    if False:
        for i in range(10):
            print('nop')
    output = target * (np.log(target) - x)
    loss = np.where(target >= 0, output, np.zeros_like(x))
    if reduction == 'batchmean':
        if len(x.shape) > 0:
            return loss.sum() / x.shape[0]
        else:
            return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss

class TestKLDivLossOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.initTestCase()
        self.op_type = 'kldiv_loss'
        self.python_api = kl_div
        x = np.random.uniform(-10, 10, self.x_shape).astype('float64')
        target = np.random.uniform(-10, 10, self.x_shape).astype('float64')
        self.attrs = {'reduction': self.reduction}
        self.inputs = {'X': x, 'Target': target}
        loss = kldiv_loss(x, target, self.reduction)
        self.outputs = {'Loss': loss.astype('float64')}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Loss', no_grad_set={'Target'})

    def initTestCase(self):
        if False:
            print('Hello World!')
        self.x_shape = (4, 5, 5)
        self.reduction = 'batchmean'

class TestKLDivLossOp2(TestKLDivLossOp):

    def initTestCase(self):
        if False:
            return 10
        self.x_shape = (3, 2, 7, 7)
        self.reduction = 'none'

class TestKLDivLossOp3(TestKLDivLossOp):

    def initTestCase(self):
        if False:
            while True:
                i = 10
        self.x_shape = (2, 3, 5, 7, 9)
        self.reduction = 'mean'

class TestKLDivLossOp4(TestKLDivLossOp):

    def initTestCase(self):
        if False:
            return 10
        self.x_shape = (5, 20)
        self.reduction = 'sum'

class TestKLDivLossDygraph(unittest.TestCase):

    def run_kl_loss(self, reduction, shape=(5, 20)):
        if False:
            while True:
                i = 10
        x = np.random.uniform(-10, 10, shape).astype('float64')
        target = np.random.uniform(-10, 10, shape).astype('float64')
        gt_loss = kldiv_loss(x, target, reduction)
        with paddle.base.dygraph.guard():
            kldiv_criterion = paddle.nn.KLDivLoss(reduction)
            pred_loss = kldiv_criterion(paddle.to_tensor(x), paddle.to_tensor(target))
            np.testing.assert_allclose(pred_loss.numpy(), gt_loss, rtol=1e-05)

    def test_kl_loss_batchmean(self):
        if False:
            i = 10
            return i + 15
        self.run_kl_loss('batchmean')

    def test_kl_loss_batchmean_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_kl_loss('batchmean', ())

    def test_kl_loss_mean(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_kl_loss('mean')

    def test_kl_loss_sum(self):
        if False:
            print('Hello World!')
        self.run_kl_loss('sum')

    def test_kl_loss_none(self):
        if False:
            print('Hello World!')
        self.run_kl_loss('none')

    def test_kl_loss_static_api(self):
        if False:
            print('Hello World!')
        with paddle_static_guard():
            input = paddle.static.data(name='input', shape=[5, 20])
            label = paddle.static.data(name='label', shape=[5, 20])
            paddle.nn.functional.kl_div(input, label)
            paddle.nn.functional.kl_div(input, label, 'sum')
            paddle.nn.functional.kl_div(input, label, 'batchmean')

class TestKLDivLossTypePromotion(unittest.TestCase):

    def test_kl_div_promotion(self):
        if False:
            while True:
                i = 10
        with paddle.base.dygraph.guard():
            x1 = paddle.rand([5, 20], dtype='float32')
            target1 = paddle.rand([5, 20], dtype='float64')
            kldiv_criterion = paddle.nn.KLDivLoss()
            pred_loss1 = kldiv_criterion(x1, target1)
            x2 = paddle.rand([5, 20], dtype='float64')
            target2 = paddle.rand([5, 20], dtype='float32')
            pred_loss2 = paddle.nn.functional.kl_div(x2, target2)
if __name__ == '__main__':
    unittest.main()