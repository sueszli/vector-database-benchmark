"""Functions to verify exported ONNX model is functionally equivalent to original PyTorch model.

ONNX Runtime is required, and is used as the ONNX backend for export verification.
"""
from __future__ import annotations
import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import Any, Callable, Collection, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, _exporter_states, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.types import Number
_ORT_PROVIDERS = ('CPUExecutionProvider',)
_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, torch.jit.ScriptModule]
_InputArgsType = Union[torch.Tensor, Tuple[Any, ...]]
_InputKwargsType = Mapping[str, Any]
_OutputsType = Union[Sequence[_NumericType], Sequence]

class OnnxBackend(enum.Enum):
    """Enum class for ONNX backend used for export verification."""
    REFERENCE = 'ONNXReferenceEvaluator'
    ONNX_RUNTIME_CPU = 'CPUExecutionProvider'
    ONNX_RUNTIME_CUDA = 'CUDAExecutionProvider'

@dataclasses.dataclass
class VerificationOptions:
    """Options for ONNX export verification.

    Attributes:
        flatten: If True, unpack nested list/tuple/dict inputs into a flattened list of
            Tensors for ONNX. Set this to False if nested structures are to be preserved
            for ONNX, which is usually the case with exporting ScriptModules. Default True.
        ignore_none: Whether to ignore None type in torch output, which is usually the
            case with tracing. Set this to False, if torch output should keep None type,
            which is usually the case with exporting ScriptModules. Default to True.
        check_shape: Whether to check the shapes between PyTorch and ONNX Runtime outputs
            are exactly the same. Set this to False to allow output shape broadcasting.
            Default to True.
        check_dtype: Whether to check the dtypes between PyTorch and ONNX Runtime outputs
            are consistent. Default to True.
        backend: ONNX backend for verification. Default to OnnxBackend.ONNX_RUNTIME_CPU.
        rtol: relative tolerance in comparison between ONNX and PyTorch outputs.
        atol: absolute tolerance in comparison between ONNX and PyTorch outputs.
        remained_onnx_input_idx: If provided, only the specified inputs will be passed
            to the ONNX model. Supply a list when there are unused inputs in the model.
            Since unused inputs will be removed in the exported ONNX model, supplying
            all inputs will cause an error on unexpected inputs. This parameter tells
            the verifier which inputs to pass into the ONNX model.
        acceptable_error_percentage: acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.
    """
    flatten: bool = True
    ignore_none: bool = True
    check_shape: bool = True
    check_dtype: bool = True
    backend: OnnxBackend = OnnxBackend.ONNX_RUNTIME_CPU
    rtol: float = 0.001
    atol: float = 1e-07
    remained_onnx_input_idx: Optional[Sequence[int]] = None
    acceptable_error_percentage: Optional[float] = None

@_beartype.beartype
def _flatten_tuples(elem):
    if False:
        i = 10
        return i + 15
    flattened = []
    for t in elem:
        if isinstance(t, tuple):
            flattened.extend(_flatten_tuples(t))
        else:
            flattened.append(t)
    return flattened

def _to_numpy(elem) -> Union[list, np.ndarray]:
    if False:
        print('Hello World!')
    if isinstance(elem, torch.Tensor):
        if elem.requires_grad:
            return elem.detach().cpu().numpy()
        else:
            return elem.cpu().numpy()
    elif isinstance(elem, (list, tuple)):
        return [_to_numpy(inp) for inp in elem]
    elif isinstance(elem, (bool, int, float)):
        return np.array(elem)
    elif isinstance(elem, dict):
        flattened = []
        for k in elem:
            flattened.extend([_to_numpy(k), _to_numpy(elem[k])])
        return flattened
    return elem

@_beartype.beartype
def _inline_flatten_list(inputs, res_list) -> list:
    if False:
        return 10
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else _inline_flatten_list(i, res_list)
    return res_list

@_beartype.beartype
def _unpack_to_numpy(values, cast_onnx_accepted=True) -> list:
    if False:
        return 10
    value_unpacked = []
    for value in values:
        value_unpacked.extend(utils.unpack_quantized_tensor(value, cast_onnx_accepted=cast_onnx_accepted))
    return [_to_numpy(v) for v in value_unpacked]

@_beartype.beartype
def _run_onnx(onnx_session, inputs) -> _OutputsType:
    if False:
        print('Hello World!')
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):
        kw_inputs = inputs[-1]
        inputs = inputs[:-1]
    inputs = _unpack_to_numpy(_flatten_tuples(inputs))
    ort_inputs = {}
    for (input_name, input) in kw_inputs.items():
        ort_inputs[input_name] = _to_numpy(input)
    inputs = _to_numpy(inputs)
    if hasattr(onnx_session, 'get_inputs'):
        input_names = [i.name for i in onnx_session.get_inputs()]
    elif hasattr(onnx_session, 'input_names'):
        input_names = onnx_session.input_names
    else:
        raise ValueError(f'Unknown ONNX backend type: {type(onnx_session)}.')
    for (i, input) in enumerate(inputs):
        if i == len(input_names) or input_names[i] in ort_inputs:
            raise ValueError(f'got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}. input names: {input_names}.')
        ort_inputs[input_names[i]] = input
    onnx_outs = onnx_session.run(None, ort_inputs)
    return onnx_outs

@_beartype.beartype
def _ort_session(model: Union[str, io.BytesIO], ort_providers: Sequence[str]=_ORT_PROVIDERS):
    if False:
        i = 10
        return i + 15
    try:
        import onnxruntime
    except ImportError as e:
        raise ImportError('onnxruntime is required for export verification.') from e
    if ort_providers is None:
        ort_providers = _ORT_PROVIDERS
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3
    ort_session = onnxruntime.InferenceSession(model if isinstance(model, str) else model.getvalue(), session_options, providers=ort_providers)
    return ort_session

@_beartype.beartype
def _onnx_reference_evaluator_session(model: Union[str, io.BytesIO]):
    if False:
        i = 10
        return i + 15
    try:
        import onnx
        from onnx import reference as onnx_reference
    except ImportError as exc:
        raise ImportError('onnx >= 1.13 is required for reference evaluator.') from exc
    proto = onnx.load(model) if isinstance(model, str) else onnx.load_model_from_string(model.getvalue())
    onnx_session = onnx_reference.ReferenceEvaluator(proto)
    return onnx_session

@_beartype.beartype
def _onnx_backend_session(model: Union[str, io.BytesIO], backend: OnnxBackend):
    if False:
        print('Hello World!')
    if backend == OnnxBackend.REFERENCE:
        onnx_session = _onnx_reference_evaluator_session(model)
    elif backend in {OnnxBackend.ONNX_RUNTIME_CPU, OnnxBackend.ONNX_RUNTIME_CUDA}:
        onnx_session = _ort_session(model, (backend.value,))
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    return onnx_session

@_beartype.beartype
def _compare_onnx_pytorch_outputs_in_np(onnx_outs: _OutputsType, pt_outs: _OutputsType, options: VerificationOptions):
    if False:
        i = 10
        return i + 15
    assert len(onnx_outs) == len(pt_outs), f'Number of outputs differ ONNX runtime: ({len(onnx_outs)}) PyTorch: ({len(pt_outs)})'
    acceptable_error_percentage = options.acceptable_error_percentage
    if acceptable_error_percentage and (acceptable_error_percentage > 1.0 or acceptable_error_percentage < 0.0):
        raise ValueError('If set, acceptable_error_percentage should be between 0.0 and 1.0')
    for (ort_out, pt_out) in zip(onnx_outs, pt_outs):
        try:
            if not options.check_shape:
                (ort_out, pt_out) = np.broadcast_arrays(ort_out, pt_out)
            torch.testing.assert_close(ort_out, pt_out, rtol=options.rtol, atol=options.atol, check_dtype=options.check_dtype, equal_nan=True)
        except AssertionError as e:
            if acceptable_error_percentage:
                error_percentage = 1 - np.sum(np.isclose(ort_out, pt_out, rtol=options.rtol, atol=options.atol)) / np.prod(ort_out.shape)
                if error_percentage <= acceptable_error_percentage:
                    warnings.warn(f'Suppressed AssertionError:\n{e}.\nError percentage {error_percentage} within acceptable range {acceptable_error_percentage}.')
                    continue
            if ort_out.dtype == np.uint8 or ort_out.dtype == np.int8:
                warnings.warn('ONNX output is quantized')
            if pt_out.dtype == np.uint8 or pt_out.dtype == np.int8:
                warnings.warn('PyTorch output is quantized')
            raise

@_beartype.beartype
def _compare_onnx_pytorch_outputs(onnx_outs: _OutputsType, pt_outs: Any, options: VerificationOptions):
    if False:
        i = 10
        return i + 15
    '\n    Compare ONNX and PyTorch outputs.\n\n    Args:\n        onnx_outs: outputs from ONNX backend.\n        pt_outs: outputs from PyTorch.\n        options: options for verification.\n\n    Raises:\n        AssertionError: if outputs from ONNX model and PyTorch model are not\n            equal up to specified precision.\n        ValueError: if arguments provided are invalid.\n    '
    if options.ignore_none:
        (pt_outs, _) = torch.jit._flatten(pt_outs)
    else:
        pt_outs = _inline_flatten_list([pt_outs], [])
    pt_outs_np = _unpack_to_numpy(pt_outs, cast_onnx_accepted=False)
    onnx_outs = _inline_flatten_list(onnx_outs, [])
    _compare_onnx_pytorch_outputs_in_np(onnx_outs, pt_outs_np, options)

@_beartype.beartype
def _prepare_input_for_pytorch(args, kwargs):
    if False:
        return 10
    'Prepare input for PyTorch model execution.\n\n    Any future changes/formatting to the input before dispatching to the PyTorch\n    model should be made in this function.\n\n    Args:\n        args: positional arguments for PyTorch model forward method.\n        kwargs: keyword arguments for PyTorch model forward method.\n\n    Returns:\n        args: positional arguments for PyTorch model forward method.\n        kwargs: keyword arguments for PyTorch model forward method.\n    '
    if isinstance(args, (torch.Tensor, dict)):
        args = (args,)
    args = copy.deepcopy(args)
    if kwargs:
        kwargs = copy.deepcopy(kwargs)
    else:
        kwargs = {}
    return (args, kwargs)

@_beartype.beartype
def _prepare_input_for_export(args, kwargs):
    if False:
        return 10
    'Prepare input for ONNX model export.\n\n    Any future changes/formatting to the input before dispatching to the\n    :func:`torch.onnx.export` api should be made in this function.\n\n    Args:\n        args: positional arguments for PyTorch model forward method.\n        kwargs: keyword arguments for PyTorch model forward method.\n\n    Returns:\n        onnx_inputs: positional arguments for ONNX model export, as `args` in\n            :func:`torch.onnx.export`.\n    '
    (args, kwargs) = _prepare_input_for_pytorch(args, kwargs)
    if not kwargs and len(args) > 0 and isinstance(args[-1], dict):
        onnx_inputs = args + ({},)
    elif kwargs:
        onnx_inputs = args + (kwargs,)
    else:
        onnx_inputs = args
    return onnx_inputs

@_beartype.beartype
def _prepare_input_for_onnx(args, kwargs, remained_onnx_input_idx: Optional[Sequence[int]], flatten: bool):
    if False:
        for i in range(10):
            print('nop')
    'Prepare input for ONNX model execution in ONNX backend.\n\n    Any future changes/formatting to the input before dispatching to the ONNX backend\n    run should be made in this function.\n\n    Args:\n        args: positional arguments for PyTorch model forward method.\n        kwargs: keyword arguments for PyTorch model forward method.\n        remained_onnx_input_idx: indices of inputs to be used for ONNX model execution.\n        flatten: whether to flatten the input before dispatching to the ONNX model execution.\n\n    Returns:\n        onnx_inputs: positional arguments for ONNX model execution in ONNX backend.\n    '
    onnx_inputs = _prepare_input_for_export(args, kwargs)
    if flatten:
        (onnx_inputs, _) = torch.jit._flatten(onnx_inputs)
    elif onnx_inputs and onnx_inputs[-1] == {}:
        onnx_inputs = onnx_inputs[:-1]
    if remained_onnx_input_idx is not None:
        return [onnx_inputs[i] for i in remained_onnx_input_idx]
    else:
        return onnx_inputs

@_beartype.beartype
def _try_clone_model(model):
    if False:
        i = 10
        return i + 15
    'Used for preserving original model in case forward mutates model states.'
    try:
        return copy.deepcopy(model)
    except Exception:
        warnings.warn('Failed to clone model. Model state might be mutated during verification.')
        return model

@_beartype.beartype
def _compare_onnx_pytorch_model(pt_model: _ModelType, onnx_model_f: Union[str, io.BytesIO], input_args: _InputArgsType, input_kwargs: Optional[_InputKwargsType], additional_test_inputs: Optional[Sequence[_InputArgsType]], options: VerificationOptions):
    if False:
        print('Hello World!')
    'Compare outputs from ONNX model runs with outputs from PyTorch model runs.\n\n    Args:\n        pt_model: PyTorch model.\n        onnx_model_f: ONNX model file path or file-like object.\n        input_args: positional arguments for PyTorch model forward method.\n        input_kwargs: keyword arguments for PyTorch model forward method.\n        additional_test_inputs: additional positional arguments for PyTorch model\n            forward method.\n        options: options for verification.\n\n    Raises:\n        AssertionError: if outputs from ONNX model and PyTorch model are not\n            equal up to specified precision.\n    '
    onnx_session = _onnx_backend_session(onnx_model_f, options.backend)

    @_beartype.beartype
    def compare_onnx_pytorch_model_with_input(input_args, input_kwargs):
        if False:
            print('Hello World!')
        (pt_args, pt_kwargs) = _prepare_input_for_pytorch(input_args, input_kwargs)
        pt_model_copy = _try_clone_model(pt_model)
        pt_outs = pt_model_copy(*pt_args, **pt_kwargs)
        onnx_inputs = _prepare_input_for_onnx(input_args, input_kwargs, options.remained_onnx_input_idx, options.flatten)
        onnx_outs = _run_onnx(onnx_session, onnx_inputs)
        _compare_onnx_pytorch_outputs(onnx_outs=onnx_outs, pt_outs=pt_outs, options=options)
    compare_onnx_pytorch_model_with_input(input_args, input_kwargs)
    if additional_test_inputs:
        for test_input_args in additional_test_inputs:
            compare_onnx_pytorch_model_with_input(test_input_args, {})

class _GraphDiff:
    """A class to represent the difference between two graphs."""

    @_beartype.beartype
    def __init__(self, graph_a: _C.Graph, graph_b: _C.Graph):
        if False:
            while True:
                i = 10
        'Construct a _GraphDiff object.\n\n        Args:\n            graph_a (_C.Graph): First graph to compare.\n            graph_b (_C.Graph): Second graph to compare.\n        '
        self.graph_a = graph_a
        self.graph_b = graph_b

    @_beartype.beartype
    def __str__(self):
        if False:
            return 10
        'See function :func:`diff_report`.'
        return self.diff_report()

    @_beartype.beartype
    def _indent(self, lines: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join(['\t' + line for line in lines.splitlines()])

    @_beartype.beartype
    def diff_report(self) -> str:
        if False:
            while True:
                i = 10
        'Return a string representation of the graph difference.\n\n        The report shows the first pair of nodes that diverges. It also shows the source\n        location of the pair of nodes.\n\n        Returns:\n            graph_diff_report (str): A string representation of the graph difference.\n        '
        graph_a = self.graph_a
        graph_b = self.graph_b
        graph_a_str = str(graph_a)
        graph_b_str = str(graph_b)
        if graph_a_str == graph_b_str:
            return ''
        graph_diff = difflib.ndiff(graph_a_str.splitlines(True), graph_b_str.splitlines(True))
        graph_diff_report = ['Graph diff:', self._indent(''.join(graph_diff))]
        for (node_a, node_b) in itertools.zip_longest(graph_a.nodes(), graph_b.nodes()):
            if str(node_a) != str(node_b):
                graph_diff_report.append('First diverging operator:')
                node_diff = difflib.ndiff(str(node_a).splitlines(True), str(node_b).splitlines(True))
                source_printout = ['node diff:', self._indent(''.join(node_diff))]
                stack_a = node_a.sourceRange() if node_a else None
                if stack_a:
                    source_printout.extend(['Former source location:', self._indent(str(stack_a))])
                stack_b = node_b.sourceRange() if node_b else None
                if stack_b:
                    source_printout.extend(['Latter source location:', self._indent(str(stack_b))])
                graph_diff_report.extend(source_printout)
                break
        return '\n'.join(graph_diff_report)

@_beartype.beartype
def _check_graph_diff(model: Union[torch.nn.Module, torch.jit.ScriptModule], test_input_groups: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]], export_options: _experimental.ExportOptions, model_to_graph_func: Callable[[torch.nn.Module, Tuple[Any, ...], Mapping[str, Any], _experimental.ExportOptions], _C.Graph]) -> str:
    if False:
        print('Hello World!')
    'Check if graph produced by `model_to_graph_func` is the same across `test_input_groups`.\n\n    Args:\n        model: See :func:`check_export_model_diff`.\n        test_input_groups: See :func:`check_export_model_diff`.\n        export_options: See :func:`check_export_model_diff`.\n        model_to_graph_func: A function to convert a PyTorch model to a JIT IR graph.\n\n    Returns:\n        graph_diff_report (str): A string representation of the graph difference.\n    '
    if len(test_input_groups) < 2:
        raise ValueError('Need at least two groups of test inputs to compare.')
    ref_jit_graph = None
    for (args, kwargs) in test_input_groups:
        jit_graph = model_to_graph_func(model, args, kwargs, export_options)
        if ref_jit_graph is None:
            ref_jit_graph = jit_graph
            continue
        graph_diff_report = _GraphDiff(ref_jit_graph, jit_graph).diff_report()
        if graph_diff_report:
            return graph_diff_report
    return ''

@_beartype.beartype
def _traced_graph_from_model(model: Union[torch.nn.Module, torch.jit.ScriptModule], args: Tuple[Any, ...], kwargs: Mapping[str, Any], export_options: _experimental.ExportOptions) -> _C.Graph:
    if False:
        i = 10
        return i + 15
    'As part of the ONNX export steps, create a traced JIT graph from a PyTorch model.\n\n    Args:\n        model: See :func:`check_export_model_diff`.\n        args: See :func:`check_export_model_diff`.\n        kwargs: See :func:`check_export_model_diff`.\n        export_options: See :func:`check_export_model_diff`.\n\n    Returns:\n        jit_graph (_C.Graph): A traced JIT graph.\n    '
    training = export_options.training
    verbose = export_options.verbose
    with utils.exporter_context(model, training, verbose):
        export_inputs = _prepare_input_for_export(args, kwargs)
        model = utils._pre_trace_quant_model(model, export_inputs)
        (jit_graph, _, _, _) = utils._create_jit_graph(model, export_inputs)
        return jit_graph

@_beartype.beartype
def _onnx_graph_from_model(model: Union[torch.nn.Module, torch.jit.ScriptModule], args: Tuple[Any, ...], kwargs: Mapping[str, Any], export_options: _experimental.ExportOptions) -> _C.Graph:
    if False:
        return 10
    'As part of the ONNX export steps, export an ONNX JIT graph from a PyTorch model.\n\n    Args:\n        model: See :func:`check_export_model_diff`.\n        args: See :func:`check_export_model_diff`.\n        kwargs: See :func:`check_export_model_diff`.\n        export_options: See :func:`check_export_model_diff`.\n\n    Returns:\n        onnx_graph (_C.Graph): An ONNX JIT graph.\n    '
    opset_version = export_options.opset_version
    operator_export_type = export_options.operator_export_type
    export_modules_as_functions = export_options.export_modules_as_functions
    training = export_options.training
    verbose = export_options.verbose
    dynamic_axes = export_options.dynamic_axes
    input_names = export_options.input_names
    output_names = export_options.output_names
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET
    utils._setup_trace_module_map(model, export_modules_as_functions)
    if not operator_export_type:
        if _C_onnx._CAFFE2_ATEN_FALLBACK:
            operator_export_type = _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = _C_onnx.OperatorExportTypes.ONNX
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type
    with utils.exporter_context(model, training, verbose):
        do_constant_folding = utils._decide_constant_folding(export_options.do_constant_folding, operator_export_type, training)
        if dynamic_axes is None:
            dynamic_axes = {}
        utils._validate_dynamic_axes(dynamic_axes, model, input_names, output_names)
        export_inputs = _prepare_input_for_export(args, kwargs)
        export_inputs = utils._decide_input_format(model, export_inputs)
        (onnx_graph, _, _) = utils._model_to_graph(model, export_inputs, verbose, input_names, output_names, operator_export_type, do_constant_folding, training=training, dynamic_axes=dynamic_axes)
        return onnx_graph

@_beartype.beartype
def _onnx_graph_from_aten_graph(graph: torch.Graph, export_options: _experimental.ExportOptions, params_dict: Optional[Dict[str, Any]]=None) -> Tuple[torch.Graph, Dict[str, Any]]:
    if False:
        print('Hello World!')
    if params_dict is None:
        params_dict = {}
    operator_export_type = export_options.operator_export_type
    dynamic_axes = export_options.dynamic_axes or {}
    input_names = export_options.input_names
    training = export_options.training
    do_constant_folding = export_options.do_constant_folding
    opset_version = export_options.opset_version or _constants.ONNX_DEFAULT_OPSET
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type
    do_constant_folding = utils._decide_constant_folding(do_constant_folding, operator_export_type, training)
    graph = graph.copy()
    graph = utils._optimize_graph(graph, operator_export_type, params_dict=params_dict, dynamic_axes=dynamic_axes, input_names=input_names)
    if training is None or training == _C_onnx.TrainingMode.EVAL:
        params_dict = torch._C._jit_pass_onnx_eval_peephole(graph, params_dict)
    if do_constant_folding and opset_version >= _constants.ONNX_CONSTANT_FOLDING_MIN_OPSET:
        params_dict = _C._jit_pass_onnx_constant_fold(graph, params_dict, opset_version)
        _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    if GLOBALS.onnx_shape_inference:
        _C._jit_pass_onnx_graph_shape_type_inference(graph, params_dict, opset_version)
    params_dict = _C._jit_pass_onnx_eliminate_unused_items(graph, params_dict)
    if opset_version < 9:
        _C._jit_pass_onnx_cast_all_constant_to_floating(graph)
    params_dict = _C._jit_pass_filter_non_tensor_arguments(params_dict)
    _C._jit_decay_packed_param_input_types(graph)
    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    if export_options.verbose:
        print('ONNX graph: ', graph)
    return (graph, params_dict)

@_beartype.beartype
def _onnx_proto_from_onnx_graph(onnx_graph: torch.Graph, export_options: _experimental.ExportOptions, params_dict: Dict[str, Any]) -> Tuple[bytes, Mapping[str, bytes]]:
    if False:
        i = 10
        return i + 15
    opset_version = export_options.opset_version or _constants.ONNX_DEFAULT_OPSET
    dynamic_axes = export_options.dynamic_axes or {}
    operator_export_type = export_options.operator_export_type
    val_keep_init_as_ip = utils._decide_keep_init_as_input(export_options.keep_initializers_as_inputs, operator_export_type, opset_version)
    val_add_node_names = utils._decide_add_node_names(True, operator_export_type)
    custom_opsets = export_options.custom_opsets or {}
    (proto, export_map, _, _) = onnx_graph._export_onnx(params_dict, opset_version, dynamic_axes, False, operator_export_type, not export_options.verbose, val_keep_init_as_ip, custom_opsets, val_add_node_names, '', {})
    return (proto, export_map)

@_beartype.beartype
def check_export_model_diff(model: Union[torch.nn.Module, torch.jit.ScriptModule], test_input_groups: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]], export_options: Optional[_experimental.ExportOptions]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Verify exported model discrepancy between different groups of inputs.\n\n    A graph is exported for each group of inputs. The exported graphs are then compared\n    to each other, and discrepancies of first pair of nodes are reported. This function\n    first checks the jit graph. If no discrepancies were found, it then checks the onnx\n    graph.\n\n    Unless otherwise specified, the jit/ONNX graph is expected to be the same, regardless\n    of the inputs used for exporting. A discrepancy implies the graph exported is\n    not accurate when run on other groups of inputs, which will typically results in\n    runtime errors or mismatching output.\n\n    Args:\n        model (torch.nn.Module or torch.jit.ScriptModule): The model to be exported.\n        test_input_groups (Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]]): A sequence\n            of input groups to be used to export the model. Each input group is a pair of\n            (args, kwargs).\n        export_options (_experimental.ExportOptions, optional): An _experimental.ExportOptions\n            object that controls the export behavior.\n\n    Returns:\n        str: A string containing the diff of the exported models.\n    '
    export_options = _experimental.ExportOptions() if export_options is None else export_options
    jit_diff_report = _check_graph_diff(model, test_input_groups, export_options, _traced_graph_from_model)
    if jit_diff_report:
        return jit_diff_report
    return _check_graph_diff(model, test_input_groups, export_options, _onnx_graph_from_model)

@_beartype.beartype
def verify(model: _ModelType, input_args: _InputArgsType, input_kwargs: Optional[_InputKwargsType]=None, do_constant_folding: bool=True, dynamic_axes: Optional[Mapping[str, Union[Mapping[int, str], Mapping[str, Sequence[int]]]]]=None, input_names: Optional[Sequence[str]]=None, output_names: Optional[Sequence[str]]=None, training: _C_onnx.TrainingMode=_C_onnx.TrainingMode.EVAL, opset_version: Optional[int]=None, keep_initializers_as_inputs: bool=True, verbose: bool=False, fixed_batch_size: bool=False, use_external_data: bool=False, additional_test_inputs: Optional[Sequence[_InputArgsType]]=None, options: Optional[VerificationOptions]=None):
    if False:
        i = 10
        return i + 15
    'Verify model export to ONNX against original PyTorch model.\n\n    Args:\n        model (torch.nn.Module or torch.jit.ScriptModule): See :func:`torch.onnx.export`.\n        input_args (tuple): See :func:`torch.onnx.export`.\n        input_kwargs (dict): See :func:`torch.onnx.export`.\n        do_constant_folding (bool, optional): See :func:`torch.onnx.export`.\n        dynamic_axes (dict, optional): See :func:`torch.onnx.export`.\n        input_names (list, optional): See :func:`torch.onnx.export`.\n        output_names (list, optional): See :func:`torch.onnx.export`.\n        training (torch.onnx.TrainingMode): See :func:`torch.onnx.export`.\n        opset_version (int, optional): See :func:`torch.onnx.export`.\n        keep_initializers_as_inputs (bool, optional): See :func:`torch.onnx.export`.\n        verbose (bool, optional): See :func:`torch.onnx.export`.\n        fixed_batch_size (bool, optional): Legacy argument, used only by rnn test cases.\n        use_external_data (bool, optional): Explicitly specify whether to export the\n            model with external data.\n        additional_test_inputs (list, optional): List of tuples. Each tuple is a group of\n            input arguments to test. Currently only *args are supported.\n        options (_VerificationOptions, optional): A _VerificationOptions object that\n            controls the verification behavior.\n\n    Raises:\n        AssertionError: if outputs from ONNX model and PyTorch model are not\n            equal up to specified precision.\n        ValueError: if arguments provided are invalid.\n    '
    if options is None:
        options = VerificationOptions()
    if training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    with torch.no_grad(), contextlib.ExitStack() as stack:
        model_f: Union[str, io.BytesIO] = io.BytesIO()
        if use_external_data:
            tmpdir_path = stack.enter_context(tempfile.TemporaryDirectory())
            model_f = os.path.join(tmpdir_path, 'model.onnx')
        inputs_for_export = _prepare_input_for_export(input_args, input_kwargs)
        model_copy = _try_clone_model(model)
        utils._export(model, inputs_for_export, model_f, opset_version=opset_version, do_constant_folding=do_constant_folding, keep_initializers_as_inputs=keep_initializers_as_inputs, dynamic_axes=dynamic_axes, input_names=input_names, output_names=output_names, fixed_batch_size=fixed_batch_size, training=training, verbose=verbose)
        _compare_onnx_pytorch_model(pt_model=model_copy, onnx_model_f=model_f, input_args=input_args, input_kwargs=input_kwargs, additional_test_inputs=additional_test_inputs, options=options)

@_beartype.beartype
def verify_aten_graph(graph: torch.Graph, input_args: Tuple[Any, ...], export_options: _experimental.ExportOptions, params_dict: Optional[Dict[str, Any]]=None, verification_options: Optional[VerificationOptions]=None) -> Tuple[Optional[AssertionError], torch.Graph, _OutputsType, _OutputsType]:
    if False:
        return 10
    if verification_options is None:
        verification_options = VerificationOptions()
    if params_dict is None:
        params_dict = {}
    original_jit_graph = graph
    graph = graph.copy()
    graph_inputs = list(graph.inputs())
    jit_inputs = tuple([arg for arg in input_args if arg is not None])
    weights = [params_dict[v.debugName()] for v in graph_inputs[len(jit_inputs):]]
    assert all((w is not None for w in weights))
    jit_inputs = copy.deepcopy(jit_inputs)
    jit_input_and_parameters = jit_inputs + tuple(weights)
    jit_outs = torch._C._jit_interpret_graph(graph, jit_input_and_parameters)
    if not isinstance(jit_outs, (list, tuple)):
        jit_outs = [jit_outs]
    (graph, onnx_params_dict) = _onnx_graph_from_aten_graph(graph, export_options, params_dict)
    (proto, export_map) = _onnx_proto_from_onnx_graph(graph, export_options, onnx_params_dict)
    model_f: Union[str, io.BytesIO] = io.BytesIO()
    export_type = _exporter_states.ExportTypes.PROTOBUF_FILE
    onnx_proto_utils._export_file(proto, model_f, export_type, export_map)
    try:
        new_input_names = {v.debugName() for v in graph.inputs()}
        new_input_args = []
        for (v, arg) in zip(original_jit_graph.inputs(), input_args):
            if v.debugName() in new_input_names:
                new_input_args.append(arg)
        input_args = tuple(new_input_args)
        onnx_inputs = _prepare_input_for_onnx(input_args, {}, verification_options.remained_onnx_input_idx, verification_options.flatten)
        onnx_session = _onnx_backend_session(model_f, verification_options.backend)
        onnx_outs = _run_onnx(onnx_session, onnx_inputs)
        del onnx_session
        try:
            _compare_onnx_pytorch_outputs(onnx_outs=onnx_outs, pt_outs=jit_outs, options=verification_options)
        except AssertionError as e:
            return (e, graph, jit_outs, onnx_outs)
        return (None, graph, jit_outs, onnx_outs)
    except Exception as e:
        print('Unexpected error during verification.')
        print('jit graph: ', original_jit_graph)
        print('onnx graph: ', graph)
        raise e

class GraphInfoPrettyPrinter:
    graph_info: Optional[GraphInfo]
    upper_printer: Optional[GraphInfoPrettyPrinter]
    lower_printer: Optional[GraphInfoPrettyPrinter]
    graph_str_lambdas: Mapping[int, str]
    connector_str_lambdas: Mapping[int, str]
    children_str_lambdas: Mapping[int, str]

    def __init__(self, graph_info: Optional[GraphInfo]):
        if False:
            while True:
                i = 10
        self.graph_info = graph_info
        if graph_info is not None and graph_info.upper_graph_info is not None and (graph_info.lower_graph_info is not None):
            self.upper_printer = GraphInfoPrettyPrinter(graph_info.upper_graph_info)
            self.lower_printer = GraphInfoPrettyPrinter(graph_info.lower_graph_info)
        else:
            self.upper_printer = None
            self.lower_printer = None

    @_beartype.beartype
    def _total_rows(self) -> int:
        if False:
            i = 10
            return i + 15
        if self.graph_info is None:
            return 1
        if self.upper_printer and self.lower_printer:
            return self.upper_printer._total_rows() + self.lower_printer._total_rows() + 1
        return 2

    @_beartype.beartype
    def _node_count_segment_str(self) -> str:
        if False:
            return 10
        if self.graph_info is None:
            return '...'
        node_count = self.graph_info.essential_node_count()
        has_mismatch = self.graph_info.has_mismatch()
        error_node_kind = f'({self.graph_info.essential_node_kinds().pop()})' if node_count == 1 and has_mismatch else ''
        return f"{node_count} {('X' if has_mismatch else '✓')} {error_node_kind}"

    @_beartype.beartype
    def _graph_id_segment_str(self) -> str:
        if False:
            print('Hello World!')
        if self.graph_info is None:
            return ''
        return f'id: {self.graph_info.id}'

    @_beartype.beartype
    def _max_segment_columns(self) -> int:
        if False:
            while True:
                i = 10
        return max(map(len, (self._node_count_segment_str(), self._graph_id_segment_str())))

    @_beartype.beartype
    def _graph_segment_str_at_line(self, line: int) -> str:
        if False:
            while True:
                i = 10
        'Get the string representation of the graph segment at the given line.'
        if line == 0:
            result_str = self._node_count_segment_str()
            result_str += ' ' * (self._max_segment_columns() - len(result_str))
            return result_str
        if line == 1:
            result_str = self._graph_id_segment_str()
            result_str += ' ' * (self._max_segment_columns() - len(result_str))
            return result_str
        if 0 <= line < self._total_rows():
            return ' ' * self._max_segment_columns()
        return ''

    @_beartype.beartype
    def _connector_segment_str_at_line(self, line: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get the connector segment string at the given line.'
        if self.upper_printer is None and self.lower_printer is None:
            return ''
        upper_total_rows = self.upper_printer._total_rows() if self.upper_printer else 1
        lower_total_rows = self.lower_printer._total_rows() if self.lower_printer else 1
        if line == 0:
            return '  __'
        elif line < upper_total_rows + 1:
            return ' |  '
        elif line == upper_total_rows + 1:
            return ' |__'
        elif line < upper_total_rows + lower_total_rows + 1:
            return '    '
        return ''

    @_beartype.beartype
    def _children_str_at_line(self, line: int) -> str:
        if False:
            return 10
        'Get the string representation of the children at the given line.\n\n        Recursively calls `_str_at_line` on children nodes.\n        '
        if self.upper_printer is None and self.lower_printer is None:
            return ''
        upper_total_rows = self.upper_printer._total_rows() if self.upper_printer else 1
        lower_total_rows = self.lower_printer._total_rows() if self.lower_printer else 1
        if 0 <= line < upper_total_rows:
            return self.upper_printer._str_at_line(line) if self.upper_printer else '...'
        elif upper_total_rows < line < upper_total_rows + lower_total_rows + 1:
            return self.lower_printer._str_at_line(line - upper_total_rows - 1) if self.lower_printer else '...'
        return ''

    @_beartype.beartype
    def _str_at_line(self, line: int) -> str:
        if False:
            return 10
        'Get the string representation of the graph at the given line.'
        return self._graph_segment_str_at_line(line) + self._connector_segment_str_at_line(line) + self._children_str_at_line(line)

    def pretty_print(self):
        if False:
            while True:
                i = 10
        if self.graph_info is None:
            print(None)
            return
        print(' Tree: '.center(80, '='))
        total_rows = self._total_rows()
        for line in range(total_rows):
            print(self._str_at_line(line).rstrip())
        if self.graph_info.has_mismatch():
            print(' Mismatch leaf subgraphs: '.center(80, '='))
            print([graph_info.id for graph_info in self.graph_info.all_mismatch_leaf_graph_info()])
            mismatch_node_kinds: Dict[str, int] = {}
            for graph_info in self.graph_info.all_mismatch_leaf_graph_info():
                node_kinds = graph_info.essential_node_kinds()
                if len(node_kinds) == 1:
                    node_kind = node_kinds.pop()
                    mismatch_node_kinds[node_kind] = mismatch_node_kinds.get(node_kind, 0) + 1
            print(' Mismatch node kinds: '.center(80, '='))
            print(mismatch_node_kinds)
        else:
            print(' No mismatch found. '.center(80, '='))

class OnnxTestCaseRepro:

    def __init__(self, repro_dir):
        if False:
            return 10
        self.repro_dir = repro_dir
        (self.proto, self.inputs, self.outputs) = onnx_proto_utils.load_test_case(repro_dir)

    @classmethod
    @_beartype.beartype
    def create_test_case_repro(cls, proto: bytes, inputs, outputs, dir: str, name: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        'Create a repro under "{dir}/test_{name}" for an ONNX test case.\n\n        The test case contains the model and the inputs/outputs data. The directory\n        structure is as follows:\n\n        dir\n        ├── test_<name>\n        │   ├── model.onnx\n        │   └── test_data_set_0\n        │       ├── input_0.pb\n        │       ├── input_1.pb\n        │       ├── output_0.pb\n        │       └── output_1.pb\n\n        Args:\n            proto: ONNX model proto.\n            inputs: Inputs to the model.\n            outputs: Outputs of the model.\n            dir: Directory to save the repro.\n            name: Name of the test case. If not specified, a name based on current time\n                will be generated.\n        Returns:\n            Path to the repro.\n        '
        if name is None:
            name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        return onnx_proto_utils.export_as_test_case(proto, _to_numpy(inputs), _to_numpy(outputs), name, dir)

    @_beartype.beartype
    def validate(self, options: VerificationOptions):
        if False:
            for i in range(10):
                print('nop')
        'Run the ONNX test case with options.backend, and compare with the expected outputs.\n\n        Args:\n            options: Options for validation.\n\n        Raise:\n            AssertionError: if outputs from options.backend and expected outputs are not\n                equal up to specified precision.\n        '
        onnx_session = _onnx_backend_session(io.BytesIO(self.proto), options.backend)
        run_outputs = onnx_session.run(None, self.inputs)
        if hasattr(onnx_session, 'get_outputs'):
            output_names = [o.name for o in onnx_session.get_outputs()]
        elif hasattr(onnx_session, 'output_names'):
            output_names = onnx_session.output_names
        else:
            raise ValueError(f'Unknown onnx session type: {type(onnx_session)}')
        expected_outs = [self.outputs[name] for name in output_names]
        _compare_onnx_pytorch_outputs_in_np(run_outputs, expected_outs, options)

@dataclasses.dataclass
class GraphInfo:
    """GraphInfo contains validation information of a TorchScript graph and its converted ONNX graph."""
    graph: torch.Graph
    input_args: Tuple[Any, ...]
    params_dict: Dict[str, Any]
    export_options: _experimental.ExportOptions = dataclasses.field(default_factory=_experimental.ExportOptions)
    mismatch_error: Optional[AssertionError] = dataclasses.field(default=None, init=False)
    pt_outs: Optional[Sequence[_NumericType]] = dataclasses.field(default=None, init=False)
    upper_graph_info: Optional[GraphInfo] = dataclasses.field(default=None, init=False)
    lower_graph_info: Optional[GraphInfo] = dataclasses.field(default=None, init=False)
    id: str = dataclasses.field(default='')
    _onnx_graph: Optional[torch.Graph] = dataclasses.field(init=False, default=None)
    _EXCLUDED_NODE_KINDS: FrozenSet[str] = frozenset({'prim::Constant', 'prim::ListConstruct', 'aten::ScalarImplicit'})

    def clear(self):
        if False:
            print('Hello World!')
        'Clear states and results of previous verification.'
        self.mismatch_error = None
        self.pt_outs = None
        self._onnx_graph = None
        self.upper_graph_info = None
        self.lower_graph_info = None

    def pretty_print_tree(self):
        if False:
            return 10
        "Pretty print `GraphInfo` tree.\n\n        Each node represents a subgraph, showing the number of nodes in the subgraph and\n        a check mark if the subgraph has output mismatch between torch and ONNX.\n\n        The id of the subgraph is shown under the node. The `GraphInfo` object for any\n        subgraph can be retrieved by calling `graph_info.find_partition(id)`.\n\n        Example::\n\n            ==================================== Tree: =====================================\n            5 X   __2 X    __1 ✓\n            id:  |  id: 0 |  id: 00\n                 |        |\n                 |        |__1 X (aten::relu)\n                 |           id: 01\n                 |\n                 |__3 X    __1 ✓\n                    id: 1 |  id: 10\n                          |\n                          |__2 X     __1 X (aten::relu)\n                             id: 11 |  id: 110\n                                    |\n                                    |__1 ✓\n                                       id: 111\n            =========================== Mismatch leaf subgraphs: ===========================\n            ['01', '110']\n            ============================= Mismatch node kinds: =============================\n            {'aten::relu': 2}\n\n        "
        GraphInfoPrettyPrinter(self).pretty_print()

    def pretty_print_mismatch(self, graph: bool=False):
        if False:
            while True:
                i = 10
        'Pretty print details of the mismatch between torch and ONNX.\n\n        Args:\n            graph: If True, print the ATen JIT graph and ONNX graph.\n        '
        print(f' Mismatch info for graph partition {self.id}: '.center(80, '='))
        if graph:
            print(' ATen JIT graph '.center(80, '='))
            print(self.graph)
            if self._onnx_graph is not None:
                print(' ONNX graph '.center(80, '='))
                print(self._onnx_graph)
        if self.has_mismatch():
            print(' Mismatch error '.center(80, '='))
            print(self.mismatch_error)
        else:
            print(' No mismatch '.center(80, '='))

    @_beartype.beartype
    def has_mismatch(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if the subgraph has output mismatch between torch and ONNX.'
        return self.mismatch_error is not None

    @_beartype.beartype
    def essential_node_count(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the number of nodes in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`.'
        return sum((1 for n in self.graph.nodes() if n.kind() not in self._EXCLUDED_NODE_KINDS))

    @_beartype.beartype
    def essential_node_kinds(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return the set of node kinds in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`.'
        return {n.kind() for n in self.graph.nodes() if n.kind() not in self._EXCLUDED_NODE_KINDS}

    @_beartype.beartype
    def all_mismatch_leaf_graph_info(self) -> List['GraphInfo']:
        if False:
            i = 10
            return i + 15
        'Return a list of all leaf `GraphInfo` objects that have mismatch.'
        if not self.has_mismatch():
            return []
        no_mismatch_children = (self.upper_graph_info is None or not self.upper_graph_info.has_mismatch()) and (self.lower_graph_info is None or not self.lower_graph_info.has_mismatch())
        if no_mismatch_children:
            return [self]
        results = []
        if self.upper_graph_info is not None:
            results += self.upper_graph_info.all_mismatch_leaf_graph_info()
        if self.lower_graph_info is not None:
            results += self.lower_graph_info.all_mismatch_leaf_graph_info()
        return results

    @_beartype.beartype
    def find_partition(self, id: str) -> Optional['GraphInfo']:
        if False:
            return 10
        'Find the `GraphInfo` object with the given id.'
        if id == self.id:
            return self
        current_length = len(self.id)
        if len(id) > current_length:
            if id[current_length] == '0' and self.upper_graph_info is not None:
                return self.upper_graph_info.find_partition(id)
            elif id[current_length] == '1' and self.lower_graph_info is not None:
                return self.lower_graph_info.find_partition(id)
        return None

    @_beartype.beartype
    def export_repro(self, repro_dir: Optional[str]=None, name: Optional[str]=None) -> str:
        if False:
            i = 10
            return i + 15
        'Export the subgraph to ONNX along with the input/output data for repro.\n\n        The repro directory will contain the following files::\n\n            dir\n            ├── test_<name>\n            │   ├── model.onnx\n            │   └── test_data_set_0\n            │       ├── input_0.pb\n            │       ├── input_1.pb\n            │       ├── output_0.pb\n            │       └── output_1.pb\n\n        Args:\n            repro_dir: The directory to export the repro files to. Defaults to current\n                working directory if None.\n            name: An optional name for the test case folder: "test_{name}".\n\n        Returns:\n            The path to the exported repro directory.\n        '
        if repro_dir is None:
            repro_dir = os.getcwd()
        repro_dir = os.path.join(repro_dir, 'onnx_debug')
        (onnx_graph, onnx_params_dict) = _onnx_graph_from_aten_graph(self.graph, self.export_options, self.params_dict)
        (proto, _) = _onnx_proto_from_onnx_graph(onnx_graph, self.export_options, onnx_params_dict)
        return OnnxTestCaseRepro.create_test_case_repro(proto, self.input_args, self.pt_outs, repro_dir, name)

    @_beartype.beartype
    def _graph_partition_pivot(self) -> int:
        if False:
            while True:
                i = 10
        'Find the pivot index to partition the graph.\n\n        The pivot is the node that splits the graph into two parts. Each part should\n        have the similar amount of nodes, excluding non essential ops, defined in\n        `_EXCLUDED_NODE_KINDS`, such as `prim::Constant`.\n        If the graph has an odd number of nodes, the upper part will have one more node.\n        If the graph does not have any node that can be partitioned, return -1.\n\n        Returns:\n            The index of the pivot node.\n        '
        included_node_indices = [i for (i, n) in enumerate(self.graph.nodes()) if n.kind() not in self._EXCLUDED_NODE_KINDS]
        half_idx = len(included_node_indices) // 2 - 1
        if half_idx >= 0 and len(included_node_indices) > half_idx:
            return included_node_indices[half_idx] + 1
        return -1

    @_beartype.beartype
    def _partition_upper_graph(self) -> torch.Graph:
        if False:
            print('Hello World!')
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()
        original_outputs = list(graph.outputs())

        def _process_bridge_value_for_upper(new_outputs: List[torch.Value], bridge_value: torch.Value) -> torch.Value:
            if False:
                return 10
            new_outputs.append(bridge_value)
            return bridge_value
        new_outputs: List[torch.Value] = []
        process_bridge_value_for_upper = functools.partial(_process_bridge_value_for_upper, new_outputs)
        (_, dropped_nodes, complete_upper_nodes_set, _) = self._partition_nodes(graph, pivot, process_bridge_value_for_upper)
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)
        for node in reversed(dropped_nodes):
            node.destroy()
        for (i, input) in reversed(list(enumerate(list(graph.inputs())))):
            if not _has_uses_by_nodes(input, complete_upper_nodes_set) and input not in new_outputs:
                try:
                    graph.eraseInput(i)
                except RuntimeError as e:
                    print(input, graph)
                    raise e
        return graph

    @_beartype.beartype
    def _partition_lower_graph(self) -> torch.Graph:
        if False:
            for i in range(10):
                print('nop')
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()
        original_outputs = list(graph.outputs())
        original_inputs = list(graph.inputs())
        new_outputs = []

        def _process_bridge_value_for_lower(graph: torch.Graph, bridge_value: torch.Value) -> torch.Value:
            if False:
                while True:
                    i = 10
            new_input = graph.addInput()
            bridge_value.replaceAllUsesWith(new_input)
            new_input.copyMetadata(bridge_value)
            return new_input
        process_bridge_value_for_lower = functools.partial(_process_bridge_value_for_lower, graph)
        (upper_nodes, lower_nodes, _, complete_lower_nodes_set) = self._partition_nodes(graph, pivot, process_bridge_value_for_lower)
        for output in original_outputs:
            if _produced_by(output, lower_nodes):
                new_outputs.append(output)
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)
        for input in original_inputs:
            if _has_uses_by_nodes(input, complete_lower_nodes_set):
                new_input = graph.addInput()
                input.replaceAllUsesWith(new_input)
                new_input.copyMetadata(input)
        for node in reversed(upper_nodes):
            if node not in complete_lower_nodes_set:
                try:
                    node.destroy()
                except RuntimeError as e:
                    print(node, graph)
                    raise e
        for _ in original_inputs:
            graph.eraseInput(0)
        return graph

    @_beartype.beartype
    def _partition_node(self, node: torch.Node, complete_upper_nodes_set: Set[torch.Node], complete_lower_nodes_set: Set[torch.Node], original_graph_outputs: Set[torch.Value], covered_bridge_values: Set[torch.Value], process_bridge_value: Callable[[torch.Value], torch.Value]):
        if False:
            while True:
                i = 10
        if node in complete_lower_nodes_set:
            return
        if _node_has_uses_by(node, complete_lower_nodes_set) and node.kind() in self._EXCLUDED_NODE_KINDS:
            complete_lower_nodes_set.update(_all_nodes([node]))
            for input in node.inputs():
                if input in covered_bridge_values:
                    continue
                self._partition_node(input.node(), complete_upper_nodes_set, complete_lower_nodes_set, original_graph_outputs, covered_bridge_values, process_bridge_value)
        else:
            for output in node.outputs():
                if output in covered_bridge_values:
                    continue
                if _has_uses_by_nodes(output, complete_lower_nodes_set) or output in original_graph_outputs:
                    covered_bridge_values.add(process_bridge_value(output))

    @_beartype.beartype
    def _partition_nodes(self, graph: torch.Graph, pivot: int, process_bridge_value: Callable[[torch.Value], torch.Value]) -> Tuple[List[torch.Node], List[torch.Node], Set[torch.Node], Set[torch.Node]]:
        if False:
            while True:
                i = 10
        nodes = list(graph.nodes())
        upper_nodes = nodes[:pivot]
        lower_nodes = nodes[pivot:]
        complete_upper_nodes_set = _all_nodes(upper_nodes)
        complete_lower_nodes_set = _all_nodes(lower_nodes)
        original_graph_outputs = set(graph.outputs())
        covered_bridge_values = set(graph.inputs())
        for node in upper_nodes:
            self._partition_node(node, complete_upper_nodes_set, complete_lower_nodes_set, original_graph_outputs, covered_bridge_values, process_bridge_value)
        return (upper_nodes, lower_nodes, complete_upper_nodes_set, complete_lower_nodes_set)

    @_beartype.beartype
    def _bridge_kwargs(self):
        if False:
            while True:
                i = 10
        pt_outs = self.pt_outs
        graph_outputs = list(self.graph.outputs())
        assert pt_outs is not None
        assert len(graph_outputs) == len(pt_outs), f'{len(graph_outputs)} vs {len(pt_outs)}\nGraph: {self.graph}'
        return {v.debugName(): o for (v, o) in zip(graph_outputs, pt_outs)}

    @_beartype.beartype
    def _args_and_params_for_partition_graph(self, graph: torch.Graph, bridge_kwargs: Mapping[str, Union[_NumericType, Sequence[_NumericType]]], full_kwargs: Mapping[str, torch.Tensor], full_params: Mapping[str, torch.Tensor]):
        if False:
            while True:
                i = 10
        input_names = [input.debugName() for input in graph.inputs()]
        args = tuple((bridge_kwargs[k] for k in input_names if k in bridge_kwargs))
        args += tuple((full_kwargs[k] for k in input_names if k in full_kwargs))
        params = {k: full_params[k] for k in input_names if k in full_params}
        assert len(args) + len(params) == len(input_names), f'{len(args)} + {len(params)} vs {len(input_names)}: {input_names}'
        return (args, params)

    @_beartype.beartype
    def verify_export(self, options: VerificationOptions) -> Tuple[Optional[AssertionError], torch.Graph, _OutputsType, _OutputsType]:
        if False:
            while True:
                i = 10
        '\n        Verify the export from TorchScript IR graph to ONNX.\n\n        Export the TorchScript IR graph to ONNX, with the inputs, parameters and export\n        options recorded in this object. Then verify the exported ONNX graph against\n        the original TorchScript IR graph under the provided verification options.\n\n        Args:\n            options: The verification options.\n\n        Returns:\n            error: The AssertionError raised during the verification. Returns None if no\n            error is raised.\n            onnx_graph: The exported ONNX graph in TorchScript IR format.\n            onnx_outs: The outputs from running exported ONNX model under the onnx\n            backend in `options`.\n            pt_outs: The outputs from running the TorchScript IR graph.\n        '
        return verify_aten_graph(self.graph, input_args=self.input_args, params_dict=self.params_dict, export_options=self.export_options, verification_options=options)

    @_beartype.beartype
    def find_mismatch(self, options: Optional[VerificationOptions]=None):
        if False:
            return 10
        '\n        Find all mismatches between the TorchScript IR graph and the exported onnx model.\n\n        Binary searches the model graph to find the minimal subgraph that exhibits the\n        mismatch. A `GraphInfo` object is created for each subgraph, recording the test\n        inputs and export options, as well as the validation results.\n\n        Args:\n            options: The verification options.\n        '
        self.clear()
        if options is None:
            options = VerificationOptions()
        if self.export_options.verbose:
            print(self.graph)
        if len(list(self.graph.outputs())) == 0:
            return
        assert len(self.input_args) + len(self.params_dict) == len(list(self.graph.inputs())), f'Number of graph inputs({len(list(self.graph.inputs()))}) does not match the provided tensor arguments({len(self.input_args)} + {len(self.params_dict)}).'
        (self.mismatch_error, self._onnx_graph, self.pt_outs, _) = self.verify_export(options)
        if self.mismatch_error is None:
            return
        if self.essential_node_count() <= 1:
            return
        full_kwargs = {k.debugName(): v for (k, v) in zip(self.graph.inputs(), self.input_args)}
        full_params = self.params_dict
        upper_graph = self._partition_upper_graph()
        (upper_args, upper_params) = self._args_and_params_for_partition_graph(upper_graph, {}, full_kwargs, full_params)
        self.upper_graph_info = GraphInfo(upper_graph, upper_args, upper_params, self.export_options, id=self.id + '0')
        self.upper_graph_info.find_mismatch(options)
        bridge_kwargs = self.upper_graph_info._bridge_kwargs()
        lower_graph = self._partition_lower_graph()
        (lower_args, lower_params) = self._args_and_params_for_partition_graph(lower_graph, bridge_kwargs, full_kwargs, full_params)
        self.lower_graph_info = GraphInfo(lower_graph, lower_args, lower_params, self.export_options, id=self.id + '1')
        self.lower_graph_info.find_mismatch(options)

@_beartype.beartype
def _all_nodes(nodes: Collection[torch.Node]) -> Set[torch.Node]:
    if False:
        return 10
    all_nodes = set(nodes)
    for n in nodes:
        for b in n.blocks():
            all_nodes.update(_all_nodes(list(b.nodes())))
    return all_nodes

@_beartype.beartype
def _has_uses_by_nodes(value: torch.Value, nodes: Collection[torch.Node]) -> bool:
    if False:
        return 10
    if any((use.user in nodes for use in value.uses())):
        return True
    return False

@_beartype.beartype
def _node_has_uses_by(node: torch.Node, nodes: Collection[torch.Node]) -> bool:
    if False:
        i = 10
        return i + 15
    for output in node.outputs():
        if _has_uses_by_nodes(output, nodes):
            return True
    return False

@_beartype.beartype
def _produced_by(value: torch.Value, nodes: Collection[torch.Node]) -> bool:
    if False:
        print('Hello World!')
    return value.node() in nodes

@_beartype.beartype
def find_mismatch(model: Union[torch.nn.Module, torch.jit.ScriptModule], input_args: Tuple[Any, ...], do_constant_folding: bool=True, training: _C_onnx.TrainingMode=_C_onnx.TrainingMode.EVAL, opset_version: Optional[int]=None, keep_initializers_as_inputs: bool=True, verbose: bool=False, options: Optional[VerificationOptions]=None) -> GraphInfo:
    if False:
        return 10
    'Find all mismatches between the original model and the exported model.\n\n    Experimental. The API is subject to change.\n\n    This tool helps debug the mismatch between the original PyTorch model and exported\n    ONNX model. It binary searches the model graph to find the minimal subgraph that\n    exhibits the mismatch.\n\n    Args:\n        model: The model to be exported.\n        input_args: The input arguments to the model.\n        do_constant_folding: Same as `do_constant_folding` in :func:`torch.onnx.export`.\n        training: Same as `training` in :func:`torch.onnx.export`.\n        opset_version: Same as `opset_version` in :func:`torch.onnx.export`.\n        keep_initializers_as_inputs: Same as `keep_initializers_as_inputs` in :func:`torch.onnx.export`.\n        verbose: Same as `verbose` in :func:`torch.onnx.export`.\n        options: The options for the mismatch verification.\n\n    Returns:\n        A GraphInfo object that contains the mismatch information.\n\n    Example::\n\n        >>> import torch\n        >>> import torch.onnx.verification\n        >>> torch.manual_seed(0)\n        >>> opset_version = 15\n        >>> # Define a custom symbolic function for aten::relu.\n        >>> # The custom symbolic function is incorrect, which will result in mismatches.\n        >>> def incorrect_relu_symbolic_function(g, self):\n        ...     return self\n        >>> torch.onnx.register_custom_op_symbolic(\n        ...     "aten::relu",\n        ...     incorrect_relu_symbolic_function,\n        ...     opset_version=opset_version,\n        ... )\n        >>> class Model(torch.nn.Module):\n        ...     def __init__(self):\n        ...         super().__init__()\n        ...         self.layers = torch.nn.Sequential(\n        ...             torch.nn.Linear(3, 4),\n        ...             torch.nn.ReLU(),\n        ...             torch.nn.Linear(4, 5),\n        ...             torch.nn.ReLU(),\n        ...             torch.nn.Linear(5, 6),\n        ...         )\n        ...     def forward(self, x):\n        ...         return self.layers(x)\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)\n        >>> graph_info = torch.onnx.verification.find_mismatch(\n        ...     Model(),\n        ...     (torch.randn(2, 3),),\n        ...     opset_version=opset_version,\n        ... )\n        ===================== Mismatch info for graph partition : ======================\n        ================================ Mismatch error ================================\n        Tensor-likes are not close!\n        Mismatched elements: 12 / 12 (100.0%)\n        Greatest absolute difference: 0.2328854203224182 at index (1, 2) (up to 1e-07 allowed)\n        Greatest relative difference: 0.699536174352349 at index (1, 3) (up to 0.001 allowed)\n        ==================================== Tree: =====================================\n        5 X   __2 X    __1 ✓\n        id:  |  id: 0 |  id: 00\n             |        |\n             |        |__1 X (aten::relu)\n             |           id: 01\n             |\n             |__3 X    __1 ✓\n                id: 1 |  id: 10\n                      |\n                      |__2 X     __1 X (aten::relu)\n                         id: 11 |  id: 110\n                                |\n                                |__1 ✓\n                                   id: 111\n        =========================== Mismatch leaf subgraphs: ===========================\n        [\'01\', \'110\']\n        ============================= Mismatch node kinds: =============================\n        {\'aten::relu\': 2}\n\n    '
    if options is None:
        options = VerificationOptions()
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET
    'From aten graph, do binary search on graph partition to find operator export discrepancy.'
    if training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    with torch.no_grad():
        inputs_for_export = _prepare_input_for_export(input_args, {})
        args = utils._decide_input_format(model, inputs_for_export)
        model = utils._pre_trace_quant_model(model, args)
        (graph, params, torch_out, module) = utils._create_jit_graph(model, args)
        params_dict = utils._get_named_param_dict(graph, params)
        utils._apply_friendly_debug_names(graph, params_dict)
        graph_info = GraphInfo(graph, input_args, params_dict, _experimental.ExportOptions(do_constant_folding=do_constant_folding, training=training, opset_version=opset_version, keep_initializers_as_inputs=keep_initializers_as_inputs, verbose=verbose))
        graph_info.find_mismatch(options)
        graph_info.pretty_print_mismatch()
        graph_info.pretty_print_tree()
        return graph_info