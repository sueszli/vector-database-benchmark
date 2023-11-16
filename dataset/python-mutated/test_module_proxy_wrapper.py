import unittest
import torch
from torch import nn
from fairseq.distributed import ModuleProxyWrapper
from .utils import objects_are_equal

class MockDDPWrapper(nn.Module):
    """A simple wrapper with an interface similar to DistributedDataParallel."""

    def __init__(self, module):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.module = module

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.module(x)

class Model(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear = nn.Linear(5, 10)
        self.xyz = 'hello'

    def forward(self, x):
        if False:
            return 10
        return self.linear(x)

    def get_xyz(self):
        if False:
            return 10
        return self.xyz

class TestModuleProxyWrapper(unittest.TestCase):

    def _get_module(self):
        if False:
            i = 10
            return i + 15
        module = Model()
        wrapped_module = MockDDPWrapper(module)
        wrapped_module = ModuleProxyWrapper(wrapped_module)
        return (wrapped_module, module)

    def test_getattr_forwarding(self):
        if False:
            while True:
                i = 10
        (wrapped_module, module) = self._get_module()
        assert module.xyz == 'hello'
        assert module.get_xyz() == 'hello'
        assert wrapped_module.xyz == 'hello'
        wrapped_module.xyz = 'world'
        assert wrapped_module.xyz == 'world'
        assert module.get_xyz() == 'hello'

    def test_state_dict(self):
        if False:
            i = 10
            return i + 15
        (wrapped_module, module) = self._get_module()
        assert objects_are_equal(wrapped_module.state_dict(), module.state_dict())

    def test_load_state_dict(self):
        if False:
            for i in range(10):
                print('nop')
        (wrapped_module, module) = self._get_module()
        wrapped_module.load_state_dict(module.state_dict())
        input = torch.rand(4, 5)
        torch.testing.assert_allclose(wrapped_module(input), module(input))

    def test_forward(self):
        if False:
            for i in range(10):
                print('nop')
        (wrapped_module, module) = self._get_module()
        input = torch.rand(4, 5)
        torch.testing.assert_allclose(wrapped_module(input), module(input))
if __name__ == '__main__':
    unittest.main()