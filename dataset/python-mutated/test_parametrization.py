import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestParametrization(JitTestCase):

    class Symmetric(nn.Module):

        def forward(self, X):
            if False:
                print('Hello World!')
            return X.triu() + X.triu(1).mT

    def test_traceable(self):
        if False:
            return 10
        'Test the jit scripting and tracing of a parametrized model.'
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, 'weight', self.Symmetric())
        x = torch.randn(3, 5)
        y = model(x)
        traced_model = torch.jit.trace_module(model, {'forward': x})
        y_hat = traced_model(x)
        self.assertEqual(y, y_hat)
        with parametrize.cached():
            y_hat = traced_model(x)
            self.assertEqual(y, y_hat)
        with self.assertRaisesRegex(RuntimeError, 'Cannot trace a model while caching'):
            with parametrize.cached():
                traced_model = torch.jit.trace_module(model, {'forward': x})

    def test_scriptable(self):
        if False:
            while True:
                i = 10
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, 'weight', self.Symmetric())
        x = torch.randn(3, 5)
        y = model(x)
        with self.assertRaises(torch.jit.Error):
            scripted_model = torch.jit.script(model)
            y_hat = scripted_model(x)
            self.assertEqual(y, y_hat)
            with parametrize.cached():
                y_hat = scripted_model(x)
                self.assertEqual(y, y_hat)
                with self.assertRaisesRegex(RuntimeError, 'Caching is not implemented'):
                    scripted_model = torch.jit.trace_module(model)