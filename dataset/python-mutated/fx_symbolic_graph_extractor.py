from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.fx
import torch.onnx
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter, io_adapter
_TORCH_METHODS_TO_PATCH: Tuple[str, ...] = ('arange', 'tensor', 'finfo', 'full', 'empty')

class ModuleExpansionTracer(torch.fx._symbolic_trace.Tracer):
    """Tracer to create ONNX-exporting friendly FX graph.

    This tracer traces models into operators. That is,
    the traced graph mostly contains call_function nodes and
    has no call_module nodes. The call_module nodes
    are problematic to the use of make_fx(...) in ONNX
    exporter.
    """

    @_beartype.beartype
    def is_leaf_module(self, module: torch.nn.Module, module_qualified_name: str) -> bool:
        if False:
            print('Hello World!')
        return False

    @_beartype.beartype
    def to_bool(self, obj: torch.fx.Proxy) -> bool:
        if False:
            i = 10
            return i + 15
        return False

def _wrap_for_symbolic_trace(target: Callable) -> Tuple[Callable, Callable]:
    if False:
        print('Hello World!')
    'This function wraps ```target`` for symbolic tracing.\n\n    This function wraps ```target``` so that its wrapper produces\n    torch.fx.Proxy in symbolic computation. The returned values are\n    the wrapper and then the original function. Per `_TORCH_METHODS_TO_PATCH`,\n    this function shall receive `torch.arange`, `torch.tensor`, etc. as inputs.\n    '

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        proxy = None

        def check_has_proxy(v):
            if False:
                print('Hello World!')
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v
        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)
        if proxy is not None:
            return proxy.tracer.create_proxy('call_function', target, args, kwargs)
        else:
            return target(*args, **kwargs)
    return (wrapper, target)

@_beartype.beartype
def _module_expansion_symbolic_trace(root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]]=None) -> torch.fx.GraphModule:
    if False:
        while True:
            i = 10
    'Trace a callable into FX graph.\n\n    When "root" is torch.nn.Module, calls to its submodule (type: torch.nn.Module) will be\n    expanded into operators (e.g., torch.matmul, torch.add, +, and -) to simplify graph\n    structure.\n    '
    patched_torch_methods = {target_name: _wrap_for_symbolic_trace(getattr(torch, target_name)) for target_name in _TORCH_METHODS_TO_PATCH}
    for (name, (wrapper, _)) in patched_torch_methods.items():
        setattr(torch, name, wrapper)
    try:
        tracer = ModuleExpansionTracer()
        graph = tracer.trace(root, concrete_args)
        name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
        return torch.fx.GraphModule(tracer.root, graph, name)
    finally:
        for (name, (_, wrapped)) in patched_torch_methods.items():
            setattr(torch, name, wrapped)

class FXSymbolicTracer(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.fx.symbolic_trace API
    Args:
        concrete_args: Inputs to be partially specialized
            It can be used to remove control flow or data structures.
            For example::
                def f(a, b):
                    if b == True:
                        return a
                    else:
                        return a*2
            FX can typically not trace through this due to the presence of control
            flow. However, we can use `concrete_args` to specialize on the value of
            `b` to trace through this::
                f = fx.symbolic_trace(f, concrete_args={'b': False})
                assert f(3, False)  == 6
            Note that although you can still pass in different values of `b`, they will be ignored.
            It can also be used to eliminate data-structure handling from
            our function. This will use pytrees to flatten your input. To avoid
            overspecializing, pass in `fx.PH` for values that shouldn't be
            specialized. For example::
                def f(x):
                    out = 0
                    for v in x.values():
                        out += v
                    return out
                f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
                assert f({'a': 1, 'b': 2, 'c': 4}) == 7
    """

    def __init__(self, concrete_args: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.concrete_args = concrete_args

    @_beartype.beartype
    def _trace_into_fx_graph_via_fx_symbolic_trace(self, model, model_args, model_kwargs) -> torch.fx.GraphModule:
        if False:
            i = 10
            return i + 15
        bind_input_step = io_adapter.BindInputStep(torch.onnx.utils.model_signature(model))
        self.input_adapter.append_step(bind_input_step)
        (_, named_args) = bind_input_step.apply(model_args, model_kwargs)
        concrete_args = {}
        for (param_name, param_value) in named_args.items():
            if isinstance(param_value, torch.Tensor):
                concrete_args[param_name] = torch.fx._symbolic_trace.PH
            else:
                concrete_args[param_name] = param_value
        merge_kwargs_step = io_adapter.MergeKwargsIntoArgsInputStep()
        self.input_adapter.append_step(merge_kwargs_step)
        return _module_expansion_symbolic_trace(model, concrete_args=concrete_args)

    def generate_fx(self, options: exporter.ResolvedExportOptions, model: Union[torch.nn.Module, Callable], model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
        if False:
            for i in range(10):
                print('nop')
        diagnostic_context = options.diagnostic_context
        graph_module = self._trace_into_fx_graph_via_fx_symbolic_trace(model, model_args, model_kwargs)
        graph_module = passes.MovePlaceholderToFront(diagnostic_context, graph_module).run()
        replace_get_attr_with_placeholder_pass = passes.ReplaceGetAttrWithPlaceholder(diagnostic_context, graph_module)
        graph_module = replace_get_attr_with_placeholder_pass.run()
        replaced_attrs = replace_get_attr_with_placeholder_pass.replaced_attrs
        append_extra_input_step = io_adapter.LiftParametersAndBuffersIntoArgsInputStep(replaced_attrs)
        self.input_adapter.append_step(append_extra_input_step)
        graph_module = passes.MovePlaceholderToFront(diagnostic_context, graph_module).run()
        graph_module.recompile()
        updated_model_args = self.input_adapter.apply(*model_args, **model_kwargs)
        return self.pre_export_passes(options, model, graph_module, updated_model_args)

    @_beartype.beartype
    def pre_export_passes(self, options: exporter.ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
        if False:
            i = 10
            return i + 15
        return exporter.common_pre_export_passes(options, original_model, fx_module, fx_module_args)