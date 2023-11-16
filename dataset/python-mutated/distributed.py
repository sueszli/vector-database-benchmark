import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable

class DistributedVariable(VariableTracker):

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        if not DistributedVariable.is_available():
            unimplemented('torch.distributed package is not available!')

    @staticmethod
    def is_available():
        if False:
            return 10
        return torch.distributed.is_available()

def is_from_local(value):
    if False:
        return 10
    if not DistributedVariable.is_available():
        return False
    from torch.distributed._tensor import DTensor
    return inspect.isfunction(value) and value is DTensor.from_local

def is_constant_pg_functions(value):
    if False:
        while True:
            i = 10
    if not DistributedVariable.is_available():
        return False
    from torch.distributed.distributed_c10d import _get_group_tag, get_process_group_ranks
    constant_processgroup_functions = [get_process_group_ranks, _get_group_tag]
    return inspect.isfunction(value) and value in constant_processgroup_functions

class PlacementClassVariable(DistributedVariable):

    def __init__(self, value, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_placement_type(value):
        if False:
            for i in range(10):
                print('nop')
        if not DistributedVariable.is_available():
            return False
        from torch.distributed._tensor.placement_types import Placement
        return type(value) is type and issubclass(value, Placement)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            return 10
        if inspect.getattr_static(self.value, '__new__', None) in (object.__new__,) and self.source:
            new_obj = object.__new__(self.value)
            var = PlacementVariable(new_obj)
            if inspect.getattr_static(self.value, '__init__', None):
                var.call_method(tx, '__init__', args, kwargs)
                return var
        return super().call_function(tx, args, kwargs)

class PlacementVariable(DistributedVariable):

    def __init__(self, value, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_placement(value):
        if False:
            i = 10
            return i + 15
        if not DistributedVariable.is_available():
            return False
        from torch.distributed._tensor.placement_types import Placement
        return isinstance(value, Placement)

    def as_python_constant(self):
        if False:
            return 10
        return self.value

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            print('Hello World!')
        from . import ConstantVariable
        allowed_methods = ['__init__', '__setattr__']
        if name in allowed_methods:
            try:
                value_type = type(self.value)
                assert inspect.getattr_static(value_type, '__getattr__', None) is None, 'no custom getattr allowed!'
                method = inspect.getattr_static(value_type, name)
            except AttributeError:
                method = None
            if method is object.__init__:
                return ConstantVariable.create(None)
            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for (k, v) in kwargs.items()}
            method(self.value, *args, **kwargs)
            return self
        return super().call_method(tx, name, args, kwargs)

class DeviceMeshVariable(DistributedVariable):

    def __init__(self, value, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_device_mesh(value):
        if False:
            print('Hello World!')
        if not DistributedVariable.is_available():
            return False
        from torch.distributed._tensor.device_mesh import DeviceMesh
        return istype(value, DeviceMesh)

    def as_python_constant(self):
        if False:
            return 10
        return self.value

    def var_getattr(self, tx, name: str) -> VariableTracker:
        if False:
            while True:
                i = 10
        if name == 'ndim':
            return ConstantVariable.create(self.value.ndim)
        return super().var_getattr(tx, name)

class ProcessGroupVariable(DistributedVariable):
    """
    We don't want a ProcessGroup object to end up in our output graph.

    But it's common for dynamo to intercept a PG that is then used to get info like
    rank() or world_size(), as well as passed to utility functions in distributed_c10d
    which desugar it into plain types like a ranklist and tag.

    For convenience and proper guarding, we construct a variable type.

    TODO: make it possible to use ProcessGroupVariable as input to simple functions
          like _expand_group without dynamo complaining about making a proxy for it.
          It is not a tensor-like type, and we don't want a proxy- but dynamo assumes
          torch library functions are dealing with tensor-like types and would have proxies
          for their args.
    TODO: should we make this inherit VT instead of UDOV? Do we want any of the default behaviors
          or just graph-break whenever one of our special cases is not hit?
    """

    def __init__(self, value, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        if False:
            while True:
                i = 10
        return self.value

    def python_type(self):
        if False:
            i = 10
            return i + 15
        return type(self.value)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            i = 10
            return i + 15
        if name == 'rank':
            return variables.ConstantVariable.create(self.value.rank())
        if name == 'size':
            return variables.ConstantVariable.create(self.value.size())
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
        if False:
            i = 10
            return i + 15
        if name in ['rank', 'size']:
            return variables.LambdaVariable(lambda *args, **kwargs: self.call_method(tx, name, args, kwargs))
        return super().var_getattr(tx, name)

    @staticmethod
    def is_process_group(value):
        if False:
            return 10
        if not DistributedVariable.is_available():
            return False
        from torch._C._distributed_c10d import ProcessGroup
        from torch.testing._internal.distributed.fake_pg import FakeProcessGroup
        return istype(value, (ProcessGroup, FakeProcessGroup))