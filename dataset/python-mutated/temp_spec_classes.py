"""This file contains temporary stubs for TensorDict, SpecDict, and ModelConfig.
This allows us to implement modules utilizing these APIs before the actual
changes land in master.
This file is to be removed once these modules are commited to master.
"""
from typing import Any

class TensorDict:

    def __init__(self, d=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if d is None:
            d = {}
        self.d = d

    def filter(self, specs: 'SpecDict') -> 'TensorDict':
        if False:
            for i in range(10):
                print('nop')
        return TensorDict({k: v for (k, v) in self.d.items() if k in specs.keys()})

    def __eq__(self, other: 'TensorDict') -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        return self.d[idx]

    def __iter__(self):
        if False:
            print('Hello World!')
        for k in self.d:
            yield k

    def flatten(self):
        if False:
            while True:
                i = 10
        return self

    def keys(self):
        if False:
            return 10
        return self.d.keys()

class SpecDict:

    def __init__(self, d=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if d is None:
            d = {}
        self.d = d

    def keys(self):
        if False:
            i = 10
            return i + 15
        return self.d.keys()

    def validate(self, spec: Any) -> bool:
        if False:
            print('Hello World!')
        return True

class ModelConfig:
    name = 'Bork'