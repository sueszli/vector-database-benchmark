import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA
aten = torch.ops.aten
const = torch.tensor(0.0)
device = 'cuda'

class TestReinplacingPassCorrectness(TestCase):

    def _test(self, f):
        if False:
            for i in range(10):
                print('nop')
        nf = torch.compile(f)
        inp = (torch.randn(4, device=device), torch.ones(2, device=device, dtype=torch.int))
        inp2 = (inp[0].clone(), inp[1].clone())
        self.assertEqual(f(*inp), nf(*inp2))
        self.assertEqual(inp, inp2)

    def test_dont_modify_live(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                i = 10
                return i + 15
            x = x.cos()
            x2 = x.index_put((y,), const)
            return (x2, x)
        self._test(f)

    def test_dont_modify_view_of_live(self):
        if False:
            return 10

        def f(x, y):
            if False:
                print('Hello World!')
            x = x.cos()
            x2 = aten.alias(x)
            x2 = x2.index_put((y,), const)
            y = x2 + x.cos()
            return y
        self._test(f)

    def test_dont_modify_input(self):
        if False:
            return 10

        def f(x, y):
            if False:
                while True:
                    i = 10
            return x.index_put((y,), const)
        self._test(f)

    def test_should_modify_inner(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x = x.cos()
            x = x.index_put((y,), const)
            return x
        self._test(f)

    def test_should_modify_input(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                while True:
                    i = 10
            x = x.index_put_((y,), const)
            return x
        self._test(f)
if __name__ == '__main__':
    if IS_LINUX and HAS_CUDA:
        run_tests()