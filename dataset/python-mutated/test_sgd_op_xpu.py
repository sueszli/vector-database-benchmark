import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op import Operator
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
from paddle.base import core

class XPUTestSgdOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'sgd'
        self.use_dynamic_create_class = False

    class TestSGDOp(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'sgd'
            self.dtype = self.in_type
            self.conf()
            w = np.random.random((self.h, self.w)).astype(self.dtype)
            g = np.random.random((self.h, self.w)).astype(self.dtype)
            lr = np.array([0.1]).astype(self.dtype)
            self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
            self.outputs = {'ParamOut': w - lr * g}

        def conf(self):
            if False:
                while True:
                    i = 10
            self.h = 102
            self.w = 105

        def test_check_output_with_place(self):
            if False:
                return 10
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestSGDOpCase8X(TestSGDOp):

        def conf(self):
            if False:
                while True:
                    i = 10
            self.h = 10
            self.w = 64
support_types = get_xpu_op_support_types('sgd')
for stype in support_types:
    create_test_class(globals(), XPUTestSgdOp, stype)

class TestSGDOpWithLargeInput(unittest.TestCase):

    def runTest(self):
        if False:
            print('Hello World!')
        data = paddle.tensor.fill_constant(shape=[1], value=128, dtype='int64')
        label = paddle.tensor.fill_constant(shape=[1, 150], value=0.5, dtype='float32')
        emb = paddle.static.nn.embedding(input=data, size=(10000, 150), dtype='float32')
        out = paddle.nn.functional.normalize(x=emb, axis=-1)
        cost = paddle.nn.functional.square_error_cost(input=out, label=label)
        avg_cost = paddle.mean(cost)
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)
        place = paddle.XPUPlace(0)
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        result = exe.run(base.default_main_program(), fetch_list=[avg_cost])

class TestSparseSGDOp(unittest.TestCase):

    def check_with_place(self, place):
        if False:
            for i in range(10):
                print('nop')
        scope = core.Scope()
        height = 10
        rows = [0, 4, 7]
        self.conf()
        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype('float32')
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0
        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, self.row_numel), 5.0).astype('float32')
        param.set(param_array, place)
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full(1, 2.0).astype('float32')
        lr.set(lr_array, place)
        sgd_op = Operator('sgd', Param='Param', Grad='Grad', ParamOut='Param', LearningRate='LearningRate')
        sgd_op.run(scope, place)
        result_array = np.array(param)
        self.assertAlmostEqual(1.0, result_array[rows[0], 0])
        self.assertAlmostEqual(3.0, result_array[rows[0], 2])
        self.assertAlmostEqual(5.0, result_array[1, 0])
        self.assertAlmostEqual(3.0, result_array[rows[1], 10])
        self.assertAlmostEqual(5.0, result_array[5, 8])
        self.assertAlmostEqual(3.0, result_array[rows[2], 1])
        self.assertAlmostEqual(-3.0, result_array[rows[2], 8])

    def test_sparse_sgd(self):
        if False:
            print('Hello World!')
        places = [core.XPUPlace(0)]
        for place in places:
            self.check_with_place(place)

    def conf(self):
        if False:
            i = 10
            return i + 15
        self.row_numel = 12

class TestSparseSGDOpCase8X(TestSparseSGDOp):

    def conf(self):
        if False:
            print('Hello World!')
        self.row_numel = 16
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()