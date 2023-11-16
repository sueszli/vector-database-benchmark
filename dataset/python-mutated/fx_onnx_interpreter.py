from __future__ import annotations
import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import onnxscript
from onnxscript.function_libs.torch_lib import graph_building as onnxscript_graph_building
import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, onnxfunction_dispatcher, op_validation, type_utils as fx_type_utils
from torch.utils import _pytree

@_beartype.beartype
def _fx_node_to_onnx_message_formatter(fn: Callable, self, node: torch.fx.Node, *args, **kwargs) -> str:
    if False:
        return 10
    return f'FX Node: {node.op}:{node.target}[name={node.name}]. '

@_beartype.beartype
def _fx_graph_to_onnx_message_formatter(fn: Callable, self, fx_graph_module: torch.fx.GraphModule, *args, **kwargs) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'FX Graph: {fx_graph_module._get_name()}. '

def _location_from_fx_stack_trace(node_stack_trace: str) -> Optional[diagnostics.infra.Location]:
    if False:
        for i in range(10):
            print('nop')
    'Extract location from FX node stack trace.\n\n    TODO(bowbao): Create fx utils module and move this function there.\n\n    Args:\n        node_stack_trace: The stack trace of the FX node. Example:\n\n            File "path/file.py", line 311, in <function>\n                <code>\n            |   File "path/file2.py", line 389, in <function>\n                <code>\n\n    Returns:\n        location: The location of the FX node.\n    '
    if 'File' not in node_stack_trace:
        return None
    lines = node_stack_trace.strip().split('\n')
    idx = 0
    while idx < len(lines) and 'File' not in lines[idx]:
        idx += 1
    if idx + 1 >= len(lines):
        return None
    pattern = re.compile('^File \\"(.+)\\", line (\\d+), in (.+)$')
    matches = pattern.match(lines[idx].strip())
    if matches:
        uri = matches.group(1)
        line_number = int(matches.group(2))
        snippet = lines[idx + 1].strip()
        return diagnostics.infra.Location(uri=uri, line=line_number, snippet=snippet)
    return None

@_beartype.beartype
def _retrieve_or_adapt_input_to_graph_set(fx_node_arg: fx_type_utils.Argument, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator):
    if False:
        return 10
    'Map FX value to TorchScript value.\n\n    When creating TorchScript graph from FX graph, we need a mapping from FX variable\n    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.\n    '
    onnx_tensor = fx_node_arg
    if isinstance(onnx_tensor, torch.fx.Node):
        return fx_name_to_onnxscript_value[onnx_tensor.name]
    if isinstance(onnx_tensor, (tuple, list)) and any((isinstance(node, torch.fx.Node) and fx_type_utils.is_torch_symbolic_type(node.meta.get('val')) for node in onnx_tensor)):
        sequence_mixed_elements: List[Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...], List[int]]] = []
        for tensor in onnx_tensor:
            if isinstance(tensor, torch.fx.Node) and fx_type_utils.is_torch_symbolic_type(tensor.meta.get('val')):
                sequence_mixed_elements.append(fx_name_to_onnxscript_value[tensor.name])
            elif isinstance(tensor, int):
                sequence_mixed_elements.append([tensor])
        with onnxscript.evaluator.default_as(tracer):
            output = onnxscript.opset18.Concat(*sequence_mixed_elements, axis=0)
        output.dtype = torch.int64
        output.shape = [len(sequence_mixed_elements)]
        return output
    elif isinstance(onnx_tensor, (tuple, list)) and all((isinstance(node, torch.fx.Node) or node is None for node in onnx_tensor)):
        sequence_elements: List[Union[Optional[onnxscript_graph_building.TorchScriptTensor], Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]] = []
        for tensor in onnx_tensor:
            sequence_elements.append(fx_name_to_onnxscript_value[tensor.name] if tensor is not None else None)
        return sequence_elements
    if isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(jit_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type())
    if isinstance(onnx_tensor, torch.device):
        return str(onnx_tensor)
    return onnx_tensor

def filter_incompatible_and_dtype_convert_kwargs(kwargs):
    if False:
        print('Hello World!')
    'Filter out kwargs that are not supported by onnxscript.'
    filtered = {}
    for (key, value) in kwargs.items():
        if key in {'layout', 'device', 'requires_grad', 'pin_memory', 'memory_format', 'implicit'}:
            continue
        if key == 'dtype':
            if value is None:
                continue
            else:
                value = int(jit_type_utils.JitScalarType.from_dtype(value).onnx_type())
        filtered[key] = value
    return filtered

@_beartype.beartype
def _fill_tensor_shape_type(onnxscript_values: Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]], name: str, expected_values: Union[fx_type_utils.META_VALUE_TYPE, List[fx_type_utils.META_VALUE_TYPE], Tuple[Optional[fx_type_utils.META_VALUE_TYPE], ...]]):
    if False:
        return 10
    'Fill the meta information of onnxscript_values with that from the fx FakeTensor.'
    if isinstance(expected_values, (list, tuple)) and (not isinstance(onnxscript_values, (list, tuple))):
        return
    (flat_onnxscript_values, _) = _pytree.tree_flatten(onnxscript_values)
    (flat_expected_values, _) = _pytree.tree_flatten(expected_values)
    for (i, (onnxscript_value, expected_value)) in enumerate(zip(flat_onnxscript_values, flat_expected_values)):
        if expected_value is None:
            continue
        elif fx_type_utils.is_torch_symbolic_type(expected_value):
            onnxscript_value.dtype = fx_type_utils.from_sym_value_to_torch_dtype(expected_value)
        elif fx_type_utils.is_torch_complex_dtype(expected_value.dtype):
            onnxscript_value.shape = (*[dim if isinstance(dim, int) else None for dim in expected_value.size()], 2)
            onnxscript_value.dtype = fx_type_utils.from_complex_to_float(expected_value.dtype)
            onnxscript_value.is_complex = True
        else:
            onnxscript_value.shape = tuple([dim if isinstance(dim, int) else None for dim in expected_value.size()])
            onnxscript_value.dtype = expected_value.dtype
        if i > 0:
            onnxscript_value.name = f'{name}_{i}'
        else:
            onnxscript_value.name = name

@_beartype.beartype
def _fill_in_default_kwargs(node: torch.fx.Node) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    if False:
        while True:
            i = 10
    'Find and Fill in the not provided kwargs with default values.'
    if hasattr(node.target, '_schema'):
        node_schema = node.target._schema
    else:
        node_schema = torch.ops.aten.sym_size.int._schema
    complete_args: List[fx_type_utils.Argument] = []
    complete_kwargs: Dict[str, fx_type_utils.Argument] = {}
    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        for (i, expected_arg) in enumerate(node_schema.arguments):
            if i < len(node.args):
                complete_args.append(node.args[i])
            elif expected_arg.name in node.kwargs:
                complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
            else:
                complete_kwargs[expected_arg.name] = expected_arg.default_value
    return (complete_args, complete_kwargs)

@_beartype.beartype
def _wrap_fx_args_as_onnxscript_args(complete_args: List[fx_type_utils.Argument], complete_kwargs: Dict[str, fx_type_utils.Argument], fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator) -> Tuple[Sequence[Optional[Union[onnxscript_graph_building.TorchScriptTensor, str, int, float, bool, list]]], Dict[str, fx_type_utils.Argument]]:
    if False:
        return 10
    'Map all FX arguments of a node to arguments in TorchScript graph.'
    onnxscript_args = tuple((_retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscript_value, tracer) for arg in complete_args))
    onnxscript_kwargs = filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)
    return (onnxscript_args, onnxscript_kwargs)

class FxOnnxInterpreter:
    """Stateless class to process FX graph Nodes and translate them into their ONNX counterparts.

    All FX nodes described by [FX Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) are supported.
    Similarly to [FX Interpreter pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter), each FX node
    must be implemented on its own method in this class.

    Each operator's implementation returns either an `onnxscript.OnnxFunction` or
    `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm. They can
    also raise RuntimeError: If there are no overloaded functions available for the given FX node.

    TODO: Convert methods to @staticmethod when the diagnostic system supports it
          DO NOT ADD NEW ATTRIBUTES TO THIS CLASS!
    """

    def __init__(self, diagnostic_context: diagnostics.DiagnosticContext):
        if False:
            return 10
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    @diagnostics.diagnose_call(diagnostics.rules.fx_node_to_onnx, diagnostic_message_formatter=_fx_node_to_onnx_message_formatter)
    def run_node(self, node, fx_graph_module: torch.fx.GraphModule, onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher, op_level_debug: bool, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]]):
        if False:
            print('Hello World!')
        'Execute a single FX node to produce its ONNX counterpart.\n\n        Args:\n            node: The FX node to be translated.\n            fx_graph_module: The FX graph module containing the node.\n            onnxfunction_dispatcher: The dispatcher to find the best matched ONNX op.\n            op_level_debug (bool): Whether to enable op level debug.\n            onnxscript_graph: The ONNX graph to be populated.\n            onnxscript_tracer: The tracer to trace the ONNX graph.\n            fx_name_to_onnxscript_value: The mapping from FX node name to ONNX Script value.\n\n        Raises:\n            RuntimeError: When a node.op is not supported.\n        '
        node_stack_trace = node.stack_trace
        if node_stack_trace:
            diagnostic = self.diagnostic_context.inflight_diagnostic(rule=diagnostics.rules.fx_node_to_onnx)
            with diagnostic.log_section(logging.INFO, 'PyTorch source information'):
                diagnostic.info('```\n%s\n```', node_stack_trace)
            location = _location_from_fx_stack_trace(node_stack_trace)
            if location is not None:
                diagnostic.with_location(location)
        if node.op == 'placeholder':
            self.placeholder(node, onnxscript_graph, fx_name_to_onnxscript_value)
        elif node.op == 'get_attr':
            self.get_attr(node, onnxscript_graph, fx_name_to_onnxscript_value, fx_graph_module)
        elif node.op == 'call_function':
            self.call_function(node, onnxscript_tracer, fx_name_to_onnxscript_value, onnxfunction_dispatcher, op_level_debug, fx_graph_module)
        elif node.op == 'call_method':
            self.call_method(node)
        elif node.op == 'call_module':
            self.call_module(node, onnxscript_graph, fx_name_to_onnxscript_value, onnxscript_tracer, fx_graph_module, onnxfunction_dispatcher, op_level_debug)
        elif node.op == 'output':
            self.output(node, onnxscript_graph, fx_name_to_onnxscript_value)
        else:
            raise RuntimeError(f'Found node type not defined in torch.fx: {node.op}')

    @_beartype.beartype
    @diagnostics.diagnose_call(diagnostics.rules.fx_graph_to_onnx, diagnostic_message_formatter=_fx_graph_to_onnx_message_formatter)
    def run(self, fx_graph_module: torch.fx.GraphModule, onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher, op_level_debug: bool, parent_onnxscript_graph: Optional[onnxscript_graph_building.TorchScriptGraph]=None) -> onnxscript_graph_building.TorchScriptGraph:
        if False:
            i = 10
            return i + 15
        'Analyze all FX nodes and trigger their ONNX translation.\n\n        Args:\n            fx_graph_module: FX graph module to be translated.\n            onnxfunction_dispatcher: ONNX function dispatcher.\n            op_level_debug: Whether to enable op-level debug.\n            parent_onnxscript_graph: The parent TorchScript graph. Must be provided if\n                `fx_graph_module` is a submodule. If not provided,\n                `fx_graph_module` is assumed to be the root module.\n        '
        diagnostic = self.diagnostic_context.inflight_diagnostic()
        with diagnostic.log_section(logging.DEBUG, 'FX Graph:'):
            diagnostic.debug('```\n%s\n```', diagnostics.LazyString(fx_graph_module.print_readable, False))
        if parent_onnxscript_graph is not None:
            onnx_meta: Optional[_pass.GraphModuleOnnxMeta] = fx_graph_module.meta.get('onnx')
            if onnx_meta is None:
                raise RuntimeError(f'ONNX meta is not found in submodule {fx_graph_module._get_name()}. Only submodules produced by `Modularize` pass is supported in ONNX export.')
            onnx_domain = onnx_meta.package_info.to_onnx_domain_string()
        else:
            onnx_domain = None
        onnxscript_graph = onnxscript_graph_building.TorchScriptGraph(parent_onnxscript_graph, domain_name=onnx_domain)
        onnxscript_tracer = onnxscript_graph_building.TorchScriptTracingEvaluator(onnxscript_graph)
        fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]] = {}
        with torch.utils._mode_utils.no_dispatch():
            for node in fx_graph_module.graph.nodes:
                self.run_node(node, fx_graph_module, onnxfunction_dispatcher, op_level_debug, onnxscript_graph, onnxscript_tracer, fx_name_to_onnxscript_value)
        with diagnostic.log_section(logging.DEBUG, 'ONNX Graph:'):
            diagnostic.debug('```\n%s\n```', onnxscript_graph.torch_graph)
        return onnxscript_graph

    @_beartype.beartype
    def placeholder(self, node: torch.fx.Node, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]]):
        if False:
            i = 10
            return i + 15
        fake_tensor = node.meta.get('val', None)
        if fake_tensor is None:
            output = onnxscript_graph.add_input(input_name=None)
        elif isinstance(fake_tensor, torch.Tensor):
            if fx_type_utils.is_torch_complex_dtype(fake_tensor.dtype):
                fake_tensor = torch.view_as_real(fake_tensor)
            output = onnxscript_graph.add_input(input_name=node.name, shape=fake_tensor.shape, dtype=fake_tensor.dtype)
        elif fx_type_utils.is_torch_symbolic_type(fake_tensor):
            output = onnxscript_graph.add_input(input_name=node.name, shape=[], dtype=fx_type_utils.from_sym_value_to_torch_dtype(fake_tensor))
        else:
            raise RuntimeError(f"Unsupported type(node.meta['val']) for placeholder: {type(fake_tensor)}")
        assert output is not None, f'Node creates None with target={node.target} and name={node.name}'
        assert isinstance(output, onnxscript_graph_building.TorchScriptTensor)
        assert isinstance(output, onnxscript.tensor.Tensor)
        fx_name_to_onnxscript_value[node.name] = output

    @_beartype.beartype
    def call_function(self, node: torch.fx.Node, onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]], onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher, op_level_debug: bool, fx_graph_module: torch.fx.GraphModule):
        if False:
            for i in range(10):
                print('nop')
        if node.target == operator.getitem and isinstance(fx_name_to_onnxscript_value[node.args[0].name], tuple):
            onnx_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]
            index = node.args[1]
            output = onnx_tensor_tuple[index]
            assert output is not None, f'Node creates None with target={node.target} and name={node.name}'
            assert isinstance(output, (onnxscript_graph_building.TorchScriptTensor, tuple)), type(output)
            fx_name_to_onnxscript_value[node.name] = output
            return
        (fx_args, fx_kwargs) = _fill_in_default_kwargs(node)
        (onnx_args, onnx_kwargs) = _wrap_fx_args_as_onnxscript_args(fx_args, fx_kwargs, fx_name_to_onnxscript_value, onnxscript_tracer)
        symbolic_fn = onnxfunction_dispatcher.dispatch(node=node, onnx_args=onnx_args, onnx_kwargs=onnx_kwargs, diagnostic_context=self.diagnostic_context)
        with onnxscript.evaluator.default_as(onnxscript_tracer):
            output: Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]] = symbolic_fn(*onnx_args, **onnx_kwargs)
        assert output is not None, f'Node creates None with target={node.target}, name={node.name}, args={onnx_args}, kwargs={onnx_kwargs}'
        _fill_tensor_shape_type(output, node.name, node.meta['val'])
        assert isinstance(output, (onnxscript_graph_building.TorchScriptTensor, tuple)), type(output)
        if op_level_debug and node.target != torch.ops.aten.sym_size and (not isinstance(node.target, types.BuiltinFunctionType)):
            op_validation.validate_op_between_ort_torch(self.diagnostic_context, node, symbolic_fn, fx_args, fx_kwargs, fx_graph_module)
        fx_name_to_onnxscript_value[node.name] = output

    @_beartype.beartype
    def output(self, node: torch.fx.Node, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]]):
        if False:
            return 10
        if isinstance(node.args[0], torch.fx.Node):
            onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]
            onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        else:
            (flat_args, _) = _pytree.tree_flatten(node.args[0])
            for arg in flat_args:
                assert isinstance(arg, torch.fx.Node), f'arg must be a torch.fx.Node, not {type(arg)}'
                onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[arg.name]
                onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)

    @_beartype.beartype
    def call_method(self, node: torch.fx.Node):
        if False:
            print('Hello World!')
        raise RuntimeError('call_method is not supported yet.')

    @_beartype.beartype
    def call_module(self, node: torch.fx.Node, parent_onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator, root_fx_graph_module: torch.fx.GraphModule, onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher, op_level_debug: bool) -> None:
        if False:
            return 10
        "Export a fx.GraphModule submodule to ONNXScript graph.\n\n        The export process specifically targets `call_module` nodes that are created by\n        the exporter's `Modularize` pass. Each `call_module` node has an associated fx.GraphModule\n        by `node.target` underneath the root fx.GraphModule. These `call_module` nodes are exported as ONNX\n        function nodes. The related `sub_module` is then exported as an ONNX model local function,\n        which is represented by another `TorchScriptGraph`. This `TorchScriptGraph` sets the current\n        `onnxscript_graph` as its parent.\n\n        Args:\n            node: The call_module node in the FX graph that represents the submodule call.\n            parent_onnxscript_graph: The parent ONNXScript graph to which the ONNX function and\n                function node belong.\n            fx_name_to_onnxscript_value: The mapping from FX node name to ONNXScript value.\n            tracer: The tracer used to trace the ONNXScript graph.\n            root_fx_graph_module: The root FX module.\n            onnxfunction_dispatcher: The dispatcher.\n            op_level_debug: Whether to enable op-level debug.\n        "
        assert isinstance(node.target, str), f'node.target must be a str, not {type(node.target)} for node {node}.'
        sub_module = root_fx_graph_module.get_submodule(node.target)
        assert isinstance(sub_module, torch.fx.GraphModule), f'sub_module must be a torch.fx.GraphModule, not {type(sub_module)} for node {node}.'
        sub_onnxscript_graph = self.run(sub_module, onnxfunction_dispatcher, op_level_debug, parent_onnxscript_graph)
        (onnx_args, _) = _wrap_fx_args_as_onnxscript_args(list(node.args), {}, fx_name_to_onnxscript_value, tracer)
        unique_module_name = f'{sub_module._get_name()}_{node.target}'
        outputs: Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]] = parent_onnxscript_graph.add_module_call(unique_module_name, sub_onnxscript_graph, onnx_args)
        assert isinstance(outputs, (onnxscript_graph_building.TorchScriptTensor, tuple)), f'Unexpected outputs type {type(outputs)} for node {node}.'
        _fill_tensor_shape_type(outputs, node.name, node.meta['val'])
        fx_name_to_onnxscript_value[node.name] = outputs

    @_beartype.beartype
    def get_attr(self, node: torch.fx.Node, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]], fx_graph_module: torch.fx.GraphModule):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(node.target, str), f'node.target {node.target} is not a str.'
        attr_tensor = getattr(fx_graph_module, node.target)
        assert isinstance(attr_tensor, torch.Tensor), f'{attr_tensor} is not a tensor.'
        input_ = onnxscript_graph.add_initializer(name=node.target.replace('/', '.'), value=attr_tensor)
        assert isinstance(input_, onnxscript_graph_building.TorchScriptTensor)
        assert isinstance(input_, onnxscript.tensor.Tensor)
        fx_name_to_onnxscript_value[node.name] = input_