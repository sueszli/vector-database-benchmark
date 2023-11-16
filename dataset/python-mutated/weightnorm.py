import jittor as jt
from jittor import nn

def _weight_norm(v, g, dim):
    if False:
        i = 10
        return i + 15
    return v * (g / jt.norm(v, 2, dim, keepdim=True))

class WeightNorm(object):

    def __init__(self, name: str, dim: int) -> None:
        if False:
            return 10
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: nn.Module):
        if False:
            i = 10
            return i + 15
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int):
        if False:
            print('Hello World!')
        if hasattr(module, '__fhook2__') and isinstance(module.__fhook2__, WeightNorm):
            raise RuntimeError('Cannot register two weight_norm hooks on the same parameter {}'.format(name))
        if dim is None:
            dim = -1
        fn = WeightNorm(name, dim)
        weight = getattr(module, name)
        delattr(module, name)
        module.__setattr__(name + '_g', jt.norm(weight, 2, dim, keepdim=True).detach())
        module.__setattr__(name + '_v', weight.detach())
        setattr(module, name, fn.compute_weight(module))
        module.register_pre_forward_hook(fn)
        return fn

    def remove(self, module: nn.Module) -> None:
        if False:
            while True:
                i = 10
        weight = self.compute_weight(module)
        delattr(module, self.name)
        delattr(module, self.name + '_g')
        delattr(module, self.name + '_v')
        setattr(module, self.name, weight.detach())

    def __call__(self, module: nn.Module, inputs) -> None:
        if False:
            for i in range(10):
                print('nop')
        setattr(module, self.name, self.compute_weight(module))

def weight_norm(module, name, dim):
    if False:
        i = 10
        return i + 15
    " Add a module weight normalization.\n\n    :param module: input model.\n    :param name: name of the assigned parameter.\n    :param dim: which dim to carry out weightnorm.\n\n    Example::\n\n    class jt_module(jt.nn.Module):\n        def __init__(self, weight):\n            super().__init__()\n            self.linear = jt.array(weight)\n\n        def execute(self, x):\n            return jt.matmul(self.linear, x)\n    \n    jm = jt_module(weight)\n    weight_norm(jm, 'linear', -1)\n    \n    "
    WeightNorm.apply(module, name, dim)
    return module

def remove_weight_norm(module, name: str='weight'):
    if False:
        return 10
    if hasattr(module, '__fhook2__') and isinstance(module.__fhook2__, WeightNorm):
        delattr(module, '__fhook2__')
        return module
    raise ValueError("weight_norm of '{}' not found in {}".format(name, module))