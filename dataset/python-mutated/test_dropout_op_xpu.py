import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
from paddle import _legacy_C_ops, base
from paddle.base import Program, program_guard
paddle.enable_static()
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types

class XPUTestDropoutOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'dropout'
        self.use_dynamic_create_class = False

    class TestDropoutOp(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.init_inputs_shape()
            self.init_attrs()
            self.dtype = self.in_type
            self.op_type = 'dropout'
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.attrs = {'dropout_prob': self.dropout_prob, 'fix_seed': self.fix_seed, 'is_test': self.is_test, 'dropout_implementation': self.dropout_implementation}
            out = self.inputs['X'] * (1.0 - self.dropout_prob)
            if not self.is_test:
                mask = None
                if self.dropout_prob == 0.0:
                    mask = np.ones(self.shape).astype(self.dtype)
                elif self.dropout_prob == 1.0:
                    mask = np.zeros(self.shape).astype(self.dtype)
                self.outputs = {'Out': out, 'Mask': mask}
            else:
                self.outputs = {'Out': out}

        def init_inputs_shape(self):
            if False:
                print('Hello World!')
            self.shape = [32, 64]

        def init_attrs(self):
            if False:
                i = 10
                return i + 15
            self.__class__.no_need_check_grad = False
            self.dropout_prob = 0.0
            self.fix_seed = True
            self.is_test = False
            self.dropout_implementation = 'upscale_in_train'

        def test_check_output(self):
            if False:
                while True:
                    i = 10
            self.check_output()

        def test_check_grad_normal(self):
            if False:
                print('Hello World!')
            if hasattr(self.__class__, 'no_need_check_grad') and self.__class__.no_need_check_grad:
                return
            self.check_grad(['X'], 'Out')

    class TestDropoutOpInput1d(TestDropoutOp):

        def init_inputs_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = [2000]

    class TestDropoutOp2(TestDropoutOp):

        def init_inputs_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = [32, 64]

        def init_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dropout_prob = 1.0
            self.fix_seed = True
            self.is_test = False
            self.dropout_implementation = 'upscale_in_train'

    class TestDropoutOp3(TestDropoutOp):

        def init_inputs_shape(self):
            if False:
                print('Hello World!')
            self.shape = [32, 64, 2]

    class TestDropoutOp4(TestDropoutOp):

        def init_attrs(self):
            if False:
                while True:
                    i = 10
            self.__class__.no_need_check_grad = True
            self.dropout_prob = 0.35
            self.fix_seed = True
            self.is_test = True
            self.dropout_implementation = 'downgrade_in_infer'

    class TestDropoutOp5(TestDropoutOp):

        def init_inputs_shape(self):
            if False:
                print('Hello World!')
            self.shape = [32, 64, 3]

        def init_attrs(self):
            if False:
                i = 10
                return i + 15
            self.__class__.no_need_check_grad = True
            self.dropout_prob = 0.75
            self.fix_seed = True
            self.is_test = True
            self.dropout_implementation = 'downgrade_in_infer'

    class TestDropoutOpError(unittest.TestCase):

        def test_errors(self):
            if False:
                while True:
                    i = 10
            with program_guard(Program(), Program()):

                def test_Variable():
                    if False:
                        return 10
                    x1 = base.create_lod_tensor(np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace())
                    paddle.nn.functional.dropout(x1, p=0.5)
                self.assertRaises(TypeError, test_Variable)

                def test_dtype():
                    if False:
                        i = 10
                        return i + 15
                    x2 = paddle.static.data(name='x2', shape=[-1, 3, 4, 5, 6], dtype='int32')
                    paddle.nn.functional.dropout(x2, p=0.5)
                self.assertRaises(TypeError, test_dtype)

    class TestDropoutCAPI(unittest.TestCase):

        def setUp(self):
            if False:
                return 10
            np.random.seed(123)
            self.places = [base.CPUPlace()]
            self.places.append(base.XPUPlace(0))

        def test_dygraph(self):
            if False:
                i = 10
                return i + 15
            for place in self.places:
                with base.dygraph.guard(place):
                    input_np = np.random.random([40, 40]).astype(self.in_type)
                    result_np = input_np
                    input = base.dygraph.to_variable(input_np)
                    m = paddle.nn.Dropout(p=0.0)
                    m.eval()
                    result = m(input)
                    np.testing.assert_allclose(result.numpy(), result_np)

    class TestDropoutBackward(unittest.TestCase):

        def setUp(self):
            if False:
                print('Hello World!')
            np.random.seed(123)
            self.places = [base.CPUPlace()]
            self.places.append(base.XPUPlace(0))

        def cal_grad_upscale_train(self, mask, prob):
            if False:
                while True:
                    i = 10
            return mask.astype(self.in_type) / (1 - prob)

        def cal_grad_downscale_in_infer(self, mask):
            if False:
                for i in range(10):
                    print('nop')
            return mask.astype(self.in_type)

        def test_backward_downscale_in_infer(self):
            if False:
                for i in range(10):
                    print('nop')
            for place in self.places:
                with base.dygraph.guard(place):
                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
                    (out, mask) = _legacy_C_ops.dropout(input, 'dropout_prob', 0.5)
                    out.backward()
                    np.testing.assert_allclose(input.gradient(), self.cal_grad_downscale_in_infer(mask.numpy()))

        def test_backward_upscale_train(self):
            if False:
                for i in range(10):
                    print('nop')
            for place in self.places:
                with base.dygraph.guard(place):
                    prob = 0.5
                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
                    (out, mask) = _legacy_C_ops.dropout(input, 'dropout_prob', prob, 'dropout_implementation', 'upscale_in_train')
                    out.backward()
                    np.testing.assert_allclose(input.gradient(), self.cal_grad_upscale_train(mask.numpy(), prob))

        def test_backward_upscale_train_2(self):
            if False:
                i = 10
                return i + 15
            for place in self.places:
                with base.dygraph.guard(place):
                    prob = 0.3
                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
                    (out, mask) = _legacy_C_ops.dropout(input, 'dropout_prob', prob, 'dropout_implementation', 'upscale_in_train')
                    out.backward()
                    np.testing.assert_allclose(input.gradient(), self.cal_grad_upscale_train(mask.numpy(), prob))
support_types = get_xpu_op_support_types('dropout')
for stype in support_types:
    create_test_class(globals(), XPUTestDropoutOp, stype)
if __name__ == '__main__':
    unittest.main()