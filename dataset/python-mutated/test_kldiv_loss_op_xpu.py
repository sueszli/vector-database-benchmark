import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle.nn.functional import kl_div
paddle.enable_static()

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

class XPUTestKLDivLossOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'kldiv_loss'
        self.use_dynamic_create_class = False

    class TestKLDivLossOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.initTestCase()
            self.op_type = 'kldiv_loss'
            self.dtype = np.float32
            self.__class__.use_xpu = True
            self.python_api = kl_div
            x = np.random.uniform(-10, 10, self.x_shape).astype('float32')
            target = np.random.uniform(-10, 10, self.x_shape).astype('float32')
            self.attrs = {'reduction': self.reduction}
            self.inputs = {'X': x, 'Target': target}
            loss = kldiv_loss(x, target, self.reduction)
            self.outputs = {'Loss': loss.astype('float32')}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output(check_dygraph=True)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Loss', no_grad_set={'Target'}, check_dygraph=True)

        def initTestCase(self):
            if False:
                return 10
            self.x_shape = (4, 5, 5)
            self.reduction = 'none'

    class TestKLDivLossOp2(TestKLDivLossOp):

        def initTestCase(self):
            if False:
                return 10
            self.x_shape = (3, 2, 7, 7)
            self.reduction = 'none'

    class TestKLDivLossOp3(TestKLDivLossOp):

        def initTestCase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_shape = (2, 3, 5, 7, 9)
            self.reduction = 'none'

    class TestKLDivLossOp4(TestKLDivLossOp):

        def initTestCase(self):
            if False:
                print('Hello World!')
            self.x_shape = (5, 20)
            self.reduction = 'none'

    class TestKLDivLossDygraph(unittest.TestCase):

        def run_kl_loss(self, reduction, shape=(5, 20)):
            if False:
                return 10
            x = np.random.uniform(-10, 10, shape).astype('float32')
            target = np.random.uniform(-10, 10, shape).astype('float32')
            gt_loss = kldiv_loss(x, target, reduction)
            with paddle.base.dygraph.guard():
                kldiv_criterion = paddle.nn.KLDivLoss(reduction)
                pred_loss = kldiv_criterion(paddle.to_tensor(x), paddle.to_tensor(target))
                np.testing.assert_allclose(pred_loss.numpy(), gt_loss, rtol=1e-05)

        def test_kl_loss_none(self):
            if False:
                i = 10
                return i + 15
            self.run_kl_loss('none')

        def test_kl_loss_static_api(self):
            if False:
                print('Hello World!')
            input = paddle.static.data(name='input', shape=[5, 20])
            label = paddle.static.data(name='label', shape=[5, 20])
            paddle.nn.functional.kl_div(input, label)

    class TestKLDivLossTypePromotion(unittest.TestCase):

        def test_kl_div_promotion(self):
            if False:
                print('Hello World!')
            with paddle.base.dygraph.guard():
                x1 = paddle.rand([5, 20], dtype='float32')
                target1 = paddle.rand([5, 20], dtype='float32')
                kldiv_criterion = paddle.nn.KLDivLoss()
                pred_loss1 = kldiv_criterion(x1, target1)
                x2 = paddle.rand([5, 20], dtype='float32')
                target2 = paddle.rand([5, 20], dtype='float32')
                pred_loss2 = paddle.nn.functional.kl_div(x2, target2)
support_types = get_xpu_op_support_types('kldiv_loss')
for stype in support_types:
    create_test_class(globals(), XPUTestKLDivLossOp, stype)
if __name__ == '__main__':
    unittest.main()