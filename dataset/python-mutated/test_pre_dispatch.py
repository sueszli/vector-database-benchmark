import torch
import torch._dynamo
import torch._dynamo.test_case

class PreDispatchTests(torch._dynamo.test_case.TestCase):

    def test_no_grad_simple(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            b = a.sin()
            with torch.no_grad():
                c = b.cos()
            return b * c.sin()
        f_compiled = torch.compile(f, backend='pre_dispatch_eager')
        a_ref = torch.randn(4, requires_grad=True)
        a_test = a_ref.clone().detach().requires_grad_(True)
        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_enable_grad_and_no_grad(self):
        if False:
            return 10

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            b = a * 2
            with torch.no_grad():
                c = b * 3
                with torch.enable_grad():
                    d = c * 4
                e = d * 5
            return b + c + d + e
        f_compiled = torch.compile(f, backend='pre_dispatch_eager')
        a_ref = torch.randn(4, requires_grad=True)
        a_test = a_ref.clone().detach().requires_grad_(True)
        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_autocast_simple(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            b = a * 2
            with torch.amp.autocast(device_type='cpu'):
                c = torch.matmul(b, b)
            return b + c
        f_compiled = torch.compile(f, backend='pre_dispatch_eager')
        a_ref = torch.randn(4, device='cpu', requires_grad=True)
        a_test = a_ref.clone().detach().requires_grad_(True)
        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()