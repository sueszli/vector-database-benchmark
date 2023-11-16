from typing import Any
import torch

@torch.jit.script
class MyScriptClass:
    """Intended to be scripted."""

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.foo = x

    def set_foo(self, x):
        if False:
            i = 10
            return i + 15
        self.foo = x

@torch.jit.script
def uses_script_class(x):
    if False:
        print('Hello World!')
    'Intended to be scripted.'
    foo = MyScriptClass(x)
    return foo.foo

class IdListFeature:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.id_list = torch.ones(1, 1)

    def returns_self(self) -> 'IdListFeature':
        if False:
            while True:
                i = 10
        return IdListFeature()

class UsesIdListFeature(torch.nn.Module):

    def forward(self, feature: Any):
        if False:
            print('Hello World!')
        if isinstance(feature, IdListFeature):
            return feature.id_list
        else:
            return feature