import torch
from torch.fx._symbolic_trace import Tracer
from torch.fx.proxy import Scope
from torch.ao.nn.intrinsic import _FusedModule
from typing import List, Callable
__all__ = ['QuantizationTracer']

class ScopeContextManager(torch.fx.proxy.ScopeContextManager):

    def __init__(self, scope: Scope, current_module: torch.nn.Module, current_module_path: str):
        if False:
            i = 10
            return i + 15
        super().__init__(scope, Scope(current_module_path, type(current_module)))

class QuantizationTracer(Tracer):

    def __init__(self, skipped_module_names: List[str], skipped_module_classes: List[Callable]):
        if False:
            print('Hello World!')
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        self.scope = Scope('', None)
        self.record_stack_traces = True

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if False:
            print('Hello World!')
        return (m.__module__.startswith('torch.nn') or m.__module__.startswith('torch.ao.nn')) and (not isinstance(m, torch.nn.Sequential)) or module_qualified_name in self.skipped_module_names or type(m) in self.skipped_module_classes or isinstance(m, _FusedModule)