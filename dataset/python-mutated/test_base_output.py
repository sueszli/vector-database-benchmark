import unittest.mock
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
try:
    from diffusers.models import unet_2d
except ImportError:
    unet_2d = None

def maybe_skip(fn):
    if False:
        while True:
            i = 10
    if unet_2d is None:
        return unittest.skip('requires diffusers')(fn)
    return fn

class TestBaseOutput(torch._dynamo.test_case.TestCase):

    @maybe_skip
    def test_create(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(a):
            if False:
                i = 10
                return i + 15
            tmp = unet_2d.UNet2DOutput(a + 1)
            return tmp
        torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=1)

    @maybe_skip
    def test_assign(self):
        if False:
            i = 10
            return i + 15

        def fn(a):
            if False:
                while True:
                    i = 10
            tmp = unet_2d.UNet2DOutput(a + 1)
            tmp.sample = a + 2
            return tmp
        args = [torch.randn(10)]
        obj1 = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1.sample, obj2.sample))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def _common(self, fn, op_count):
        if False:
            while True:
                i = 10
        args = [unet_2d.UNet2DOutput(sample=torch.randn(10))]
        obj1 = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    @maybe_skip
    def test_getattr(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(obj: unet_2d.UNet2DOutput):
            if False:
                return 10
            x = obj.sample * 10
            return x
        self._common(fn, 1)

    @maybe_skip
    def test_getitem(self):
        if False:
            while True:
                i = 10

        def fn(obj: unet_2d.UNet2DOutput):
            if False:
                while True:
                    i = 10
            x = obj['sample'] * 10
            return x
        self._common(fn, 1)

    @maybe_skip
    def test_tuple(self):
        if False:
            print('Hello World!')

        def fn(obj: unet_2d.UNet2DOutput):
            if False:
                while True:
                    i = 10
            a = obj.to_tuple()
            return a[0] * 10
        self._common(fn, 1)

    @maybe_skip
    def test_index(self):
        if False:
            i = 10
            return i + 15

        def fn(obj: unet_2d.UNet2DOutput):
            if False:
                for i in range(10):
                    print('nop')
            return obj[0] * 10
        self._common(fn, 1)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()