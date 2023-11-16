from __future__ import annotations
import contextlib
from typing import Callable, Mapping, Optional
import torch
import torch._ops
import torch.fx
from torch._dispatch import python as python_dispatch
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics
from torch.onnx._internal.fx.passes import _utils

class Decompose(_pass.Transform):

    def __init__(self, diagnostic_context: diagnostics.DiagnosticContext, module: torch.fx.GraphModule, decomposition_table: Mapping[torch._ops.OpOverload, Callable], enable_dynamic_axes: bool, allow_fake_constant: Optional[bool]=False):
        if False:
            return 10
        super().__init__(diagnostic_context, module)
        self.decomposition_table = decomposition_table
        self.enable_dynamic_axes = enable_dynamic_axes
        self.allow_fake_constant = allow_fake_constant

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        if False:
            i = 10
            return i + 15
        assert not kwargs, 'kwargs is not supported in Decompose.'
        module = _utils.wrap_graph_module_for_node_meta_preservation(self.module)
        fake_mode: Optional[fake_tensor.FakeTensorMode] = self.fake_mode
        maybe_fake_args = self._maybe_fakefy_args(fake_mode, *args)
        if fake_mode is not None:
            tracing_mode = 'real'
        else:
            fake_mode = contextlib.nullcontext()
            tracing_mode = 'symbolic' if self.enable_dynamic_axes else 'fake'
        assert fake_mode is not None
        with proxy_tensor.maybe_disable_fake_tensor_mode(), python_dispatch.enable_python_dispatcher(), fake_mode:
            decomposed_module = proxy_tensor.make_fx(module, decomposition_table=self.decomposition_table, tracing_mode=tracing_mode, _allow_non_fake_inputs=True, _allow_fake_constant=self.allow_fake_constant)(*maybe_fake_args)
        _utils.replace_placeholder_name_and_target(decomposed_module, self.module)
        return decomposed_module