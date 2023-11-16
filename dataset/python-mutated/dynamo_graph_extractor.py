from __future__ import annotations
import contextlib
import functools
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.utils import _pytree as pytree

class _PyTreeExtensionContext:
    """Context manager to register PyTree extension."""
    _extensions: Dict[Type, Tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]

    def __init__(self):
        if False:
            print('Hello World!')
        self._extensions = {}
        self._register_huggingface_model_output_extension()

    def __enter__(self):
        if False:
            return 10
        for (class_type, (flatten_func, unflatten_func)) in self._extensions.items():
            pytree._register_pytree_node(class_type, flatten_func, unflatten_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        for class_type in self._extensions:
            pytree.SUPPORTED_NODES.pop(class_type)

    @_beartype.beartype
    def register_pytree_node(self, class_type: Type, flatten_func: pytree.FlattenFunc, unflatten_func: pytree.UnflattenFunc):
        if False:
            return 10
        'Register PyTree extension for a custom python type.\n\n        Args:\n            class_type: The custom python type.\n            flatten_func: The flatten function.\n            unflatten_func: The unflatten function.\n\n        Raises:\n            AssertionError: If the custom python type is already registered.\n        '
        if class_type in pytree.SUPPORTED_NODES or class_type in self._extensions:
            return
        self._extensions[class_type] = (flatten_func, unflatten_func)

    def _register_huggingface_model_output_extension(self):
        if False:
            print('Hello World!')
        try:
            from transformers import modeling_outputs
        except ImportError as e:
            return

        @_beartype.beartype
        def model_output_flatten(output: modeling_outputs.ModelOutput) -> Tuple[List[Any], pytree.Context]:
            if False:
                while True:
                    i = 10
            return (list(output.values()), (type(output), list(output.keys())))

        @_beartype.beartype
        def model_output_unflatten(values: List[Any], context: pytree.Context) -> modeling_outputs.ModelOutput:
            if False:
                while True:
                    i = 10
            (output_type, keys) = context
            return output_type(**dict(zip(keys, values)))
        named_model_output_classes = inspect.getmembers(modeling_outputs, lambda x: inspect.isclass(x) and issubclass(x, modeling_outputs.ModelOutput))
        for (_, class_type) in named_model_output_classes:
            self.register_pytree_node(class_type, model_output_flatten, model_output_unflatten)

class DynamoFlattenOutputStep(io_adapter.FlattenOutputStep):
    """Flatten nested collection and custom python types and return a flat list of elements.

    Extended from :class:`io_adapter.FlattenOutputStep` to support flattening arbitrary
    types via pytree extension. By default this supports many common user defined python
    types such as :class:`ModelOutput` from HuggingFace transformers.

    The pytree extension can be customized by passing in a ``_PyTreeExtensionContext``
    object. See :meth:`_PyTreeExtensionContext.register_pytree_node`.
    """

    def __init__(self, pytree_extension_context: Optional[_PyTreeExtensionContext]=None):
        if False:
            return 10
        super().__init__()
        self._pytree_extension_context = pytree_extension_context or _PyTreeExtensionContext()

    def apply(self, model_outputs: Any) -> Sequence[Any]:
        if False:
            for i in range(10):
                print('nop')
        'Flatten the model outputs, under the context of pytree extension.'
        with self._pytree_extension_context:
            return super().apply(model_outputs)

def _wrap_model_with_output_adapter(model: Union[torch.nn.Module, Callable], output_adapter: DynamoFlattenOutputStep) -> Callable:
    if False:
        i = 10
        return i + 15
    'Wrap model with output adapter.\n\n    This is a helper function to enable :func:`dynamo.export` on models that produce\n    custom user defined types outputs. It wraps the model with an output adapter to\n    convert the outputs to :func:`dynamo.export` compatible types, i.e. :class:`torch.Tensor`.\n\n    The adapting logic is controlled by ``output_adapter``.\n\n    Args:\n        model: PyTorch model or function.\n        output_adapter: Output adapter to apply to model output.\n    Returns:\n        Wrapped model.\n    '
    model_func = model.forward if isinstance(model, torch.nn.Module) else model

    @functools.wraps(model_func)
    def wrapped(*args, **kwargs):
        if False:
            print('Hello World!')
        return output_adapter.apply(model_func(*args, **kwargs))
    return wrapped

class DynamoExport(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.dynamo.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """

    def __init__(self, aten_graph: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.aten_graph = aten_graph or True

    def generate_fx(self, options: exporter.ResolvedExportOptions, model: Union[torch.nn.Module, Callable], model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
        if False:
            for i in range(10):
                print('nop')
        dynamo_flatten_output_step = DynamoFlattenOutputStep()
        wrapped_model = _wrap_model_with_output_adapter(model, dynamo_flatten_output_step)
        self.output_adapter.append_step(dynamo_flatten_output_step)
        fake_mode = options.fake_context.fake_mode if options.fake_context else contextlib.nullcontext()
        fx_mode = 'symbolic' if options.dynamic_shapes else 'fake'
        with fake_mode:
            (graph_module, graph_guard) = torch._dynamo.export(wrapped_model, tracing_mode=fx_mode)(*model_args, **model_kwargs)
        del graph_guard
        torch._dynamo.reset()
        self.input_adapter.append_step(io_adapter.FlattenInputWithTreeSpecValidationInputStep())
        updated_model_args = self.input_adapter.apply(*model_args, **model_kwargs)
        return self.pre_export_passes(options, model, graph_module, updated_model_args)

    @_beartype.beartype
    def pre_export_passes(self, options: exporter.ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
        if False:
            print('Hello World!')
        return exporter.common_pre_export_passes(options, original_model, fx_module, fx_module_args)