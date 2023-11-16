import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def nll_loss_1d(logs, dtype, targets, weight=None, reduction='mean', ignore_index=-100):
    if False:
        for i in range(10):
            print('nop')
    input_shape = logs.shape
    N = input_shape[0]
    C = input_shape[1]
    out = np.zeros_like(targets).astype(dtype)
    total_weight = 0
    for i in range(N):
        cur_target = targets[i]
        if cur_target == ignore_index:
            out[i] = 0
            continue
        cur_weight = weight[cur_target] if weight is not None else 1
        total_weight += cur_weight
        out[i] = -logs[i][cur_target] * cur_weight
    if reduction == 'sum':
        out = np.sum(out)
        total_weight = np.array(total_weight).astype(dtype)
        return {'Out': out, 'Total_weight': total_weight}
    elif reduction == 'mean':
        out = np.sum(out)
        if total_weight != 0:
            out /= total_weight
        total_weight = np.array(total_weight).astype(dtype)
        return {'Out': out, 'Total_weight': total_weight}
    elif reduction == 'none':
        total_weight = np.array(0).astype(dtype)
        return {'Out': out, 'Total_weight': total_weight}

def nll_loss_2d(logs, dtype, targets, weight=None, reduction='mean', ignore_index=-100):
    if False:
        return 10
    input_shape = logs.shape
    N = input_shape[0]
    H = input_shape[2]
    W = input_shape[3]
    out = np.zeros_like(targets).astype(dtype)
    total_weight = 0
    for i in range(N):
        for h in range(H):
            for w in range(W):
                cur_target = targets[i][h][w]
                if cur_target == ignore_index:
                    out[i][h][w] = 0
                    continue
                cur_weight = weight[cur_target] if weight is not None else 1
                total_weight += cur_weight
                out[i][h][w] = -logs[i][cur_target][h][w] * cur_weight
    if reduction == 'sum':
        out = np.sum(out)
        total_weight = np.array(total_weight).astype(dtype)
        return {'Out': out, 'Total_weight': total_weight}
    elif reduction == 'mean':
        out = np.sum(out)
        if total_weight != 0:
            out /= total_weight
        total_weight = np.array(total_weight).astype(dtype)
        return {'Out': out, 'Total_weight': total_weight}
    elif reduction == 'none':
        total_weight = np.array(0).astype(dtype)
        return {'Out': out, 'Total_weight': total_weight}

class XPUTestNLLLossOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'nll_loss'
        self.use_dynamic_create_class = False

    class TestNLLLossOpBase1D(XPUOpTest):
        op_type = 'nll_loss'

        def setUp(self):
            if False:
                return 10
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.set_attrs()
            self.set_inputs()
            self.inputs = {'X': self.x, 'Label': self.label}
            if self.weight is not None:
                self.inputs['Weight'] = self.weight
            self.outputs = nll_loss_1d(self.x, self.dtype, self.label, self.weight, self.attrs['reduction'])

        def set_attrs(self):
            if False:
                print('Hello World!')
            self.attrs = {'reduction': 'none'}

        def set_inputs(self):
            if False:
                print('Hello World!')
            self.class_num = 3
            x_shape = [5, self.class_num]
            label_shape = [5]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = np.random.random(self.class_num).astype(self.dtype)

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestNLLLossOpWithWeightMean1D(TestNLLLossOpBase1D):

        def set_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {'reduction': 'mean'}

    class TestNLLLossOpWithWeightSum1D(TestNLLLossOpBase1D):

        def set_attrs(self):
            if False:
                return 10
            self.attrs = {'reduction': 'sum'}

    class TestNLLLossOpWithoutWeightNone1D(TestNLLLossOpBase1D):

        def set_inputs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.class_num = 3
            x_shape = [5, self.class_num]
            label_shape = [5]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = None

        def set_attrs(self):
            if False:
                print('Hello World!')
            self.attrs = {'reduction': 'none'}

    class TestNLLLossOpWithoutWeightMean1D(TestNLLLossOpBase1D):

        def set_inputs(self):
            if False:
                return 10
            self.class_num = 3
            x_shape = [5, self.class_num]
            label_shape = [5]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = None

        def set_attrs(self):
            if False:
                i = 10
                return i + 15
            self.attrs = {'reduction': 'mean'}

    class TestNLLLossOpWithoutWeightSum1D(TestNLLLossOpBase1D):

        def set_inputs(self):
            if False:
                return 10
            self.class_num = 3
            x_shape = [5, self.class_num]
            label_shape = [5]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = None

        def set_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {'reduction': 'sum'}

    class TestNLLLossOpBase2D(XPUOpTest):
        op_type = 'nll_loss'

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.set_attrs()
            self.set_inputs()
            self.inputs = {'X': self.x, 'Label': self.label}
            if self.weight is not None:
                self.inputs['Weight'] = self.weight
            self.outputs = nll_loss_2d(self.x, self.dtype, self.label, self.weight, self.attrs['reduction'])

        def set_attrs(self):
            if False:
                i = 10
                return i + 15
            self.attrs = {'reduction': 'none'}

        def set_inputs(self):
            if False:
                return 10
            self.class_num = 3
            x_shape = [5, self.class_num, 7, 11]
            label_shape = [5, 7, 11]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = np.random.random(self.class_num).astype(self.dtype)

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestNLLLossOpWithWeightMean2D(TestNLLLossOpBase2D):

        def set_attrs(self):
            if False:
                return 10
            self.attrs = {'reduction': 'mean'}

    class TestNLLLossOpWithWeightSum2D(TestNLLLossOpBase2D):

        def set_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {'reduction': 'sum'}

    class TestNLLLossOpWithoutWeightNone2D(TestNLLLossOpBase2D):

        def set_inputs(self):
            if False:
                return 10
            self.dtype = self.in_type
            self.class_num = 3
            x_shape = [5, self.class_num, 7, 11]
            label_shape = [5, 7, 11]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = None

        def set_attrs(self):
            if False:
                return 10
            self.attrs = {'reduction': 'none'}

    class TestNLLLossOpWithoutWeightMean2D(TestNLLLossOpBase2D):

        def set_inputs(self):
            if False:
                while True:
                    i = 10
            self.dtype = self.in_type
            self.class_num = 3
            x_shape = [5, self.class_num, 7, 11]
            label_shape = [5, 7, 11]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = None

        def set_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {'reduction': 'mean'}

    class TestNLLLossOpWithoutWeightSum2D(TestNLLLossOpBase2D):

        def set_inputs(self):
            if False:
                return 10
            self.dtype = self.in_type
            self.class_num = 3
            x_shape = [5, self.class_num, 7, 11]
            label_shape = [5, 7, 11]
            self.x = np.random.random(x_shape).astype(self.dtype)
            self.label = np.random.randint(low=0, high=self.class_num, size=label_shape).astype(np.int64)
            self.weight = None

        def set_attrs(self):
            if False:
                print('Hello World!')
            self.attrs = {'reduction': 'sum'}
support_types = get_xpu_op_support_types('nll_loss')
for stype in support_types:
    create_test_class(globals(), XPUTestNLLLossOP, stype)
if __name__ == '__main__':
    unittest.main()