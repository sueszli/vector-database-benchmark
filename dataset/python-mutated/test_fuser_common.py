import torch
from torch.testing._internal.jit_utils import JitTestCase

class TestFuserCommon(JitTestCase):

    def test_autodiff_fallback(self):
        if False:
            return 10
        for rq in [True, False]:

            @torch.jit.script
            def fn(x):
                if False:
                    return 10
                return torch.max(x ** 2.0, x ** 3.0)
            x = torch.randn(5, requires_grad=not rq)
            for i in range(5):
                fn(x)
            y = fn(torch.randn(5, requires_grad=rq))
            self.assertEqual(y.requires_grad, rq)