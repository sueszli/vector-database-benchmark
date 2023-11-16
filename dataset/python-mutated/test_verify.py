import caffe2.python.onnx.backend as backend
import torch
from torch.autograd import Function
from torch.nn import Module, Parameter
from torch.testing._internal import common_utils
from verify import verify

class TestVerify(common_utils.TestCase):
    maxDiff = None

    def assertVerifyExpectFail(self, *args, **kwargs):
        if False:
            return 10
        try:
            verify(*args, **kwargs)
        except AssertionError as e:
            if str(e):
                return
            else:
                raise
        self.assertTrue(False, msg='verify() did not fail when expected to')

    def test_result_different(self):
        if False:
            return 10

        class BrokenAdd(Function):

            @staticmethod
            def symbolic(g, a, b):
                if False:
                    i = 10
                    return i + 15
                return g.op('Add', a, b)

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    i = 10
                    return i + 15
                return a.sub(b)

        class MyModel(Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return BrokenAdd().apply(x, y)
        x = torch.tensor([1, 2])
        y = torch.tensor([3, 4])
        self.assertVerifyExpectFail(MyModel(), (x, y), backend)

    def test_jumbled_params(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModel(Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = x * x
                self.param = Parameter(torch.tensor([2.0]))
                return y
        x = torch.tensor([1, 2])
        with self.assertRaisesRegex(RuntimeError, 'state_dict changed'):
            verify(MyModel(), x, backend)

    def test_dynamic_model_structure(self):
        if False:
            print('Hello World!')

        class MyModel(Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.iters = 0

            def forward(self, x):
                if False:
                    print('Hello World!')
                if self.iters % 2 == 0:
                    r = x * x
                else:
                    r = x + x
                self.iters += 1
                return r
        x = torch.tensor([1, 2])
        self.assertVerifyExpectFail(MyModel(), x, backend)

    def test_embedded_constant_difference(self):
        if False:
            print('Hello World!')

        class MyModel(Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.iters = 0

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                r = x[self.iters % 2]
                self.iters += 1
                return r
        x = torch.tensor([[1, 2], [3, 4]])
        self.assertVerifyExpectFail(MyModel(), x, backend)

    def test_explicit_test_args(self):
        if False:
            i = 10
            return i + 15

        class MyModel(Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                if x.data.sum() == 1.0:
                    return x + x
                else:
                    return x * x
        x = torch.tensor([[6, 2]])
        y = torch.tensor([[2, -1]])
        self.assertVerifyExpectFail(MyModel(), x, backend, test_args=[(y,)])
if __name__ == '__main__':
    common_utils.run_tests()