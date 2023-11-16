import dataclasses
import importlib
import logging
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
try:
    import onnx
    import onnxruntime
    from onnxruntime.capi import _pybind_state as ORTC
    importlib.import_module('onnxscript')
    import torch.onnx
    import torch.onnx._internal
    import torch.onnx._internal.diagnostics
    import torch.onnx._internal.exporter
    import torch.onnx._internal.fx.decomposition_table
    import torch.onnx._internal.fx.passes
    from torch.onnx._internal.fx import fx_onnx_interpreter
    from torch.onnx._internal.fx.type_utils import _TORCH_DTYPE_TO_NUMPY_DTYPE, _TORCH_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE
    _SUPPORT_ONNXRT = True
except ImportError:
    _SUPPORT_ONNXRT = False
__all__ = ['is_onnxrt_backend_supported', 'torch_compile_backend', 'OrtExecutionProvider', 'OrtBackendOptions', 'OrtBackend']

def is_onnxrt_backend_supported() -> bool:
    if False:
        i = 10
        return i + 15
    'Returns ``True`` if ONNX Runtime dependencies are installed and usable\n    to support TorchDynamo backend integration; ``False`` otherwise.\n\n    Example::\n\n        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)\n        >>> import torch\n        >>> if torch.onnx.is_onnxrt_backend_supported():\n        ...     @torch.compile(backend="onnxrt")\n        ...     def f(x):\n        ...             return x * x\n        ...     print(f(torch.randn(10)))\n        ... else:\n        ...     print("pip install onnx onnxscript onnxruntime")\n        ...\n    '
    return _SUPPORT_ONNXRT

def _infer_default_eps() -> Sequence[str]:
    if False:
        for i in range(10):
            print('nop')
    return ['CPUExecutionProvider']

def _nvtx_range_push(name: str):
    if False:
        print('Hello World!')
    "If PyTorch is installed with CUDA support, this starts NVTX range.\n\n    Check torch.cuda.nvtx.range_push's document for more details.\n    "
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)

def _nvtx_range_pop():
    if False:
        while True:
            i = 10
    "If PyTorch is installed with CUDA support, this terminates NVTX range.\n\n    Check torch.cuda.nvtx.range_pop's document for more details.\n    "
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()

def _get_ort_device_type(device_type: str):
    if False:
        while True:
            i = 10
    if device_type == 'cuda':
        return ORTC.OrtDevice.cuda()
    if device_type == 'cpu':
        return ORTC.OrtDevice.cpu()
    if device_type == 'ort':
        return ORTC.OrtDevice.npu()
    raise ValueError('Unsupported device type: ' + device_type)
logger = logging.getLogger(__name__)

class OrtOperatorSupport(OperatorSupport):
    """Operator support for ONNXRuntime backend.

    It has two-level of support decision. One is via support_dict and the other one
    is via extra_support_dict. The logic of using support_dict is implemented in
    OrtOperatorSupport and extra_support_dict is used by OperatorSupport.is_node_supported.
    """

    def __init__(self, support_dict: Set[Any], extra_support_dict: Dict[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(extra_support_dict)
        self._onnx_support_dict = support_dict

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        if False:
            i = 10
            return i + 15
        if node.op not in CALLABLE_NODE_OPS:
            return False
        if node.op == 'call_function' and node.target in self._onnx_support_dict:
            logger.warning('support_dict supports node.target: %s (type: %s)', node.target, type(node.target))
            return True
        logger.warning("support_dict doesn't support node.target: %s (type: %s)", node.target, type(node.target))
        if super().is_node_supported(submodules, node):
            logger.warning('extra_support_dict supports node.target: %s (type: %s)', node.target, type(node.target))
            return True
        logger.warning("extra_support_dict doesn't supports node.target: %s (type: %s)", node.target, type(node.target))
        return False

def _move_placeholder_to_front(graph_module: torch.fx.GraphModule) -> None:
    if False:
        i = 10
        return i + 15
    "\n    In torch.fx.Graph, placeholder is a special assignment node. If it's not\n    executed in the beginning, it could overwrite values computed by upstream\n    nodes.\n    "
    graph = graph_module.graph
    placeholders = []
    first_not_placeholder = None
    for node in graph.nodes:
        if node.op == 'placeholder':
            placeholders.append(node)
        if first_not_placeholder is None and node.op != 'placeholder':
            first_not_placeholder = node
    if first_not_placeholder is None:
        return
    for placeholder in placeholders:
        first_not_placeholder.prepend(placeholder)

def _replace_to_copy_with_to(fx_module: torch.fx.GraphModule) -> None:
    if False:
        i = 10
        return i + 15
    for node in fx_module.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload) and node.target.overloadpacket == torch.ops.aten._to_copy:
            is_default_layout = True
            is_on_same_device = True
            is_cast = True
            are_kwargs_supported = True
            if 'layout' in node.kwargs and node.kwargs['layout'] != torch.strided:
                is_default_layout = False
            if 'device' in node.kwargs and node.kwargs['device'] != node.args[0].meta['val'].device:
                is_on_same_device = False
            if 'dtype' not in node.kwargs:
                is_cast = False
            for kwarg in node.kwargs:
                if kwarg not in ['layout', 'device', 'dtype']:
                    are_kwargs_supported = False
            if len(node.args) == 1 and is_default_layout and is_on_same_device and is_cast and are_kwargs_supported:
                node.kwargs = {'dtype': node.kwargs['dtype']}
                node.target = torch.ops.aten.to.dtype
            else:
                raise RuntimeError(f'aten._to_copy must be replaced with other ONNX-supported aten ops.                          args={[arg.meta for arg in node.args]}, kwargs={node.kwargs}')
    fx_module.recompile()

def _infer_ep_from_device(*args) -> Tuple[str, ...]:
    if False:
        return 10
    'Return the first valid device (i.e., GPU or CPU) in argument list.'
    eps = []
    for arg in args:
        if hasattr(arg, 'device'):
            device = arg.device
            if device.type == 'cuda':
                eps.append('CUDAExecutionProvider')
            elif device.type == 'cpu':
                eps.append('CPUExecutionProvider')
    return tuple(eps)

def _extract_graph_module_inputs(graph_module: torch.fx.GraphModule) -> Tuple[Any, ...]:
    if False:
        while True:
            i = 10
    placeholders = []
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            if hasattr(node, 'meta') and 'val' in node.meta:
                assert isinstance(node.meta['val'], torch.Tensor)
            placeholders.append(node)
    return tuple(placeholders)

def _extract_graph_module_outputs(graph_module: torch.fx.GraphModule) -> Any:
    if False:
        return 10
    'Collect "val" fields from outputs metadata in this torch.fx.GraphModule.'
    for node in graph_module.graph.nodes:
        if node.op == 'output':
            return node.args[0]
    raise ValueError('No output node found in this torch.fx.GraphModule.')

def _infer_ep_from_graph_module(graph_module: torch.fx.GraphModule) -> Tuple[str, ...]:
    if False:
        while True:
            i = 10
    'Return the all valid devices (i.e., GPU or CPU) among outputs of this torch.fx.GraphModule.'
    (flattened_output_args, _) = _pytree.tree_flatten(_extract_graph_module_outputs(graph_module))
    selected_output_args = [output_arg.meta['val'] for output_arg in flattened_output_args if hasattr(output_arg, 'meta') and 'val' in output_arg.meta]
    return _infer_ep_from_device(*selected_output_args)

def _sort_eps(eps: Tuple[str, ...]) -> Tuple[str, ...]:
    if False:
        print('Hello World!')
    'Sort execution providers in eps based on pre-set priority.'

    def get_execution_provider_priority(ep: str) -> int:
        if False:
            while True:
                i = 10
        if ep == 'CPUExecutionProvider':
            return 2
        if ep == 'CUDAExecutionProvider':
            return 1
        return 0
    unique_eps = set(eps)
    return tuple(sorted(unique_eps, key=get_execution_provider_priority, reverse=True))

def _get_onnx_devices(values: Tuple[torch.Tensor, ...]) -> Tuple['ORTC.OrtDevice', ...]:
    if False:
        while True:
            i = 10
    assert all((value.device == values[0].device for value in values)), 'All values must be on the same device.'

    def _device_id_or_zero(device_id: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        return device_id or 0
    devices: Tuple['ORTC.OrtDevice', ...] = tuple((ORTC.OrtDevice(_get_ort_device_type(value.device.type), ORTC.OrtDevice.default_memory(), _device_id_or_zero(value.device.index)) for value in values))
    return devices

def _get_ortvalues_from_torch_tensors(tensors: Tuple[torch.Tensor, ...], devices: Tuple['ORTC.OrtDevice', ...]) -> Tuple[torch.Tensor, ...]:
    if False:
        print('Hello World!')
    ortvalues = ORTC.OrtValueVector()
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []
    for tensor in tensors:
        dtypes.append(_TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])
        shapes.append(tensor.size())
        data_ptrs.append(tensor.data_ptr())
    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues

def _to_real_tensor(tensor: FakeTensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    if tensor.is_sparse:
        raise ValueError('sparse tensor is not yet supported.')
    out = torch.empty(tensor.size(), dtype=tensor.dtype, device=tensor.device)
    return out

def _run_onnx_session_with_ortvaluevector(sess: 'onnxruntime.InferenceSession', input_names: Tuple[str, ...], inputs: Tuple[torch.Tensor, ...], input_devices: Tuple['ORTC.OrtDevice', ...], output_names: Tuple[str, ...], outputs: Tuple[torch.Tensor, ...], output_devices: Tuple['ORTC.OrtDevice', ...], preallocate_output: bool) -> Tuple[torch.Tensor, ...]:
    if False:
        i = 10
        return i + 15
    _nvtx_range_push('contiguous')
    inputs = tuple((a.contiguous() for a in inputs))
    _nvtx_range_pop()
    _nvtx_range_push('push_back_batch')
    ort_inputs = _get_ortvalues_from_torch_tensors(inputs, input_devices)
    if preallocate_output:
        pth_outputs = tuple((_to_real_tensor(t) if isinstance(t, FakeTensor) else t for t in outputs))
        ort_outputs = _get_ortvalues_from_torch_tensors(pth_outputs, output_devices)
    else:
        ort_outputs = ORTC.OrtValueVector()
    _nvtx_range_pop()
    _nvtx_range_push('run_with_ortvaluevector')
    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')
    sess.run_with_ortvaluevector(run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices)
    _nvtx_range_pop()
    if preallocate_output:
        return pth_outputs
    else:
        _nvtx_range_push('after run_with_ortvaluevector')
        pth_outputs = onnxruntime.training.ortmodule._utils._ortvalues_to_torch_tensor(ort_outputs)
        _nvtx_range_pop()
        return pth_outputs

def _run_onnx_session_with_fetch(sess: 'onnxruntime.InferenceSession', input_names: Tuple[str, ...], inputs: Tuple[torch.Tensor, ...], input_devices: Tuple['ORTC.OrtDevice', ...], output_names: Tuple[str, ...], outputs: Tuple[torch.Tensor, ...], output_devices: Tuple['ORTC.OrtDevice', ...], preallocate_output: bool) -> Tuple[torch.Tensor, ...]:
    if False:
        i = 10
        return i + 15
    feed = {name: onnxruntime.OrtValue.ortvalue_from_numpy(tensor.cpu().numpy()) for (name, tensor) in zip(input_names, inputs)}
    ort_outputs = sess.run(output_names, feed)
    pth_outputs = tuple((torch.from_numpy(value).to(tensor.device) for (value, tensor) in zip(ort_outputs, outputs)))
    return pth_outputs

class OrtExecutionInfoPerSession:
    """Information required to execute torch.fx.GraphModule using onnxruntime.InferenceSession"""

    def __init__(self, session: 'onnxruntime.InferenceSession', input_names: Tuple[str, ...], input_value_infos: Tuple['onnx.ValueInfoProto', ...], output_names: Tuple[str, ...], output_value_infos: Tuple['onnx.ValueInfoProto', ...], input_devices: Tuple['ORTC.OrtDevice', ...], output_devices: Tuple['ORTC.OrtDevice', ...], example_outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):
        if False:
            while True:
                i = 10
        self.session: onnxruntime.InferenceSession = session
        self.input_names: Tuple[str, ...] = input_names
        self.input_value_infos: Tuple[onnx.ValueInfoProto, ...] = input_value_infos
        self.output_names: Tuple[str, ...] = output_names
        self.output_value_infos: Tuple[onnx.ValueInfoProto, ...] = output_value_infos
        self.input_devices: Tuple['ORTC.OrtDevice', ...] = input_devices
        self.output_devices: Tuple['ORTC.OrtDevice', ...] = output_devices
        self.example_outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor] = example_outputs

    def is_supported(self, *args):
        if False:
            return 10
        if len(args) != len(self.input_value_infos):
            return False
        for (arg, value_info) in zip(args, self.input_value_infos):
            if not isinstance(arg, torch.Tensor):
                return False
            onnx_dtype = _TORCH_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE[arg.dtype]
            if onnx_dtype != value_info.type.tensor_type.elem_type:
                return False
            for (dim, onnx_dim) in zip(arg.shape, value_info.type.tensor_type.shape.dim):
                if isinstance(dim, int) and (onnx_dim.dim_value == dim or onnx_dim.dim_param):
                    continue
                elif isinstance(dim, torch.SymInt) and onnx_dim.dim_param:
                    continue
                else:
                    return False
        return True

@dataclasses.dataclass
class OrtExecutionInfoForAllGraphModules:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.execution_info_per_graph_module: Dict[torch.fx.GraphModule, List[OrtExecutionInfoPerSession]] = {}

    def search_reusable_session_execution_info(self, graph_module: torch.fx.GraphModule, *args):
        if False:
            i = 10
            return i + 15
        if graph_module not in self.execution_info_per_graph_module:
            return None
        candidates = self.execution_info_per_graph_module[graph_module]
        for candidate in candidates:
            if candidate.is_supported(*args):
                return candidate
        return None

    def cache_session_execution_info(self, graph_module: torch.fx.GraphModule, info: OrtExecutionInfoPerSession):
        if False:
            for i in range(10):
                print('nop')
        if graph_module not in self.execution_info_per_graph_module:
            self.execution_info_per_graph_module[graph_module] = [info]
        else:
            self.execution_info_per_graph_module[graph_module].append(info)
OrtExecutionProvider: TypeAlias = Union[str, Tuple[str, Mapping[str, Any]]]
'Either the name of an ONNX Runtime execution provider as a string or\na 2-tuple of the name and a dictionary of execution provider options.\n\nExamples::\n\n    >>> "CPUExecutionProvider"\n\n    >>> ("CUDAExecutionProvider", {"device_id": 3})\n\n'

@dataclasses.dataclass(frozen=True)
@compatibility(is_backward_compatible=False)
class OrtBackendOptions:
    """Options for constructing an ``OrtBackend``, the ONNX Runtime
    backend (``"onnxrt"``) for ``torch.compile``.

    Example::

        >>> @torch.compile(
        ...     backend="onnxrt",
        ...     options=torch.onnx._OrtBackendOptions(...),
        ... )
        ... def ort_function(x):
        ...     return x ** x
    """
    preferred_execution_providers: Optional[Sequence[OrtExecutionProvider]] = None
    'An optional sequence of execution providers to be prioritized ahead of any\n    execution providers that may be inferred (see ``infer_execution_providers``).\n    '
    infer_execution_providers: bool = True
    'Whether to infer an execution provider from ``torch.device`` bound to inputs or found in the graph.'
    default_execution_providers: Optional[Sequence[OrtExecutionProvider]] = None
    'The default fallback execution providers. If not specified, one will be\n    be selected based on the host environment (most likely ``"CPUExecutionProvider"``).\n    '
    preallocate_output: bool = False
    "If ``True``, allocate memory for ONNX Runtime's outputs on the PyTorch side."
    use_aot_autograd: bool = True
    "Whether to wrap the ``OrtBackend`` with TorchDynamo's aot_autograd backend\n    to support training (i.e., backward graphs are also sent to ``OrtBackend``).\n\n    Symbolic execution is used to capture the forward pass and backward passes as a single graph.\n    Then, a selected graph partition algorithm (``min_cut_rematerialization_partition``) is used\n    to split the entire graph into forward sub-graph and backward sub-graph. Finally, both\n    sub-graphs are compiled by ``OrtBackend``.\n    "
    export_options: Optional['torch.onnx.ExportOptions'] = None
    'Options for the TorchDynamo-based ONNX exporter used by the ``OrtBackend``.'
    ort_session_options: Optional['onnxruntime.SessionOptions'] = None
    'Options for the ``onnxruntime.InferenceSession`` used by the ``OrtBackend``.'

@compatibility(is_backward_compatible=False)
class OrtBackend:
    """A backend compiles (sub-)graphs in torch.fx.GraphModule to onnxruntime.InferenceSession calls.

    The compiler entry point is OrtBackend.compile, which
        1. partitions the original graph into supported sub-graphs (type: torch.fx.GraphModule) and unsupported
           sub-graphs.
        2. For each supported sub-graph, it replaces its _wrapped_call function with _ort_accelerated_call.
        3. Inside _ort_accelerated_call, it creates onnxruntime.InferenceSession and calls it to execute the sub-graph.
    """

    def __init__(self, options: Optional[OrtBackendOptions]=None):
        if False:
            i = 10
            return i + 15
        self._options: Final = OrtBackendOptions() if options is None else options
        self._resolved_onnx_exporter_options = torch.onnx._internal.exporter.ResolvedExportOptions(torch.onnx.ExportOptions() if self._options.export_options is None else self._options.export_options)
        support_dict = torch.onnx._internal.fx.decomposition_table._create_onnx_supports_op_overload_table(self._resolved_onnx_exporter_options.onnx_registry)
        extra_support_dict: Dict[str, Any] = {'getattr': None, '_operator.getitem': None}
        self._supported_ops = OrtOperatorSupport(support_dict, extra_support_dict)
        self._partitioner_cache: Dict[torch.fx.GraphModule, torch.fx.GraphModule] = {}
        self._all_ort_execution_info = OrtExecutionInfoForAllGraphModules()
        self._assert_allclose_to_baseline = False
        self.execution_count = 0
        self.run = _run_onnx_session_with_ortvaluevector if hasattr(ORTC, 'push_back_batch') else _run_onnx_session_with_fetch

    def _select_eps(self, graph_module: torch.fx.GraphModule, *args) -> Sequence[Tuple[str, Mapping[str, Any]]]:
        if False:
            return 10
        inferred_eps: Tuple[str, ...] = tuple()
        if self._options.infer_execution_providers:
            if (eps_from_args := _infer_ep_from_device(*args)):
                inferred_eps = eps_from_args
            elif (eps_from_graph_module := _infer_ep_from_graph_module(graph_module)):
                inferred_eps = eps_from_graph_module
        selected_eps = []
        for ep in (*(self._options.preferred_execution_providers or []), *_sort_eps(inferred_eps), *(self._options.default_execution_providers or _infer_default_eps())):
            if isinstance(ep, str):
                ep = (ep, {})
            elif isinstance(ep, tuple) and ep[1] is None:
                ep = (ep[0], {})
            if ep is not None and ep not in selected_eps:
                selected_eps.append(ep)
        return selected_eps

    def _ort_acclerated_call(self, graph_module: torch.fx.GraphModule, *args, **kwargs):
        if False:
            while True:
                i = 10
        'This function replaces GraphModule._wrapped_call in compiled model.\n\n        The _wrapped_call is the underlying implementation of forward method. Replacing\n        it means we delegate the computation to _ort_acclerated_call and therefore\n        onnxruntime.InferenceSession.\n        '
        cached_execution_info_per_session = self._all_ort_execution_info.search_reusable_session_execution_info(graph_module, *args)
        if cached_execution_info_per_session:
            onnx_session = cached_execution_info_per_session.session
            input_names = cached_execution_info_per_session.input_names
            output_names = cached_execution_info_per_session.output_names
            input_devices = cached_execution_info_per_session.input_devices
            output_devices = cached_execution_info_per_session.output_devices
            prim_outputs = cached_execution_info_per_session.example_outputs
        else:
            graph_module = torch.onnx._internal.fx.passes.MovePlaceholderToFront(self._resolved_onnx_exporter_options.diagnostic_context, graph_module).run()
            if self._resolved_onnx_exporter_options.dynamic_shapes:
                self.preallocate_output = False
                extracted_outputs = _extract_graph_module_outputs(graph_module)

                def maybe_map_to_meta_val(value):
                    if False:
                        while True:
                            i = 10
                    if hasattr(value, 'meta') and 'val' in value.meta:
                        return value.meta['val']
                    else:
                        return value
                prim_outputs = _pytree.tree_map(maybe_map_to_meta_val, extracted_outputs)
            else:
                try:
                    prim_outputs = FakeTensorProp(graph_module).propagate(*args, **kwargs)
                except Exception:
                    logger.warning('FakeTensorProb failed for %s', graph_module)
                    self.preallocate_output = False
                    raise
            fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(diagnostic_context=self._resolved_onnx_exporter_options.diagnostic_context)
            graph_module = torch.onnx._internal.fx.passes.InsertTypePromotion(self._resolved_onnx_exporter_options.diagnostic_context, graph_module).run()
            exported = fx_interpreter.run(fx_graph_module=graph_module, onnxfunction_dispatcher=self._resolved_onnx_exporter_options.onnxfunction_dispatcher, op_level_debug=self._resolved_onnx_exporter_options.op_level_debug)
            onnx_model = exported.to_model_proto(opset_version=self._resolved_onnx_exporter_options.onnx_registry.opset_version)
            onnx_session = onnxruntime.InferenceSession(path_or_bytes=onnx_model.SerializeToString(), sess_options=self._options.ort_session_options, providers=self._select_eps(graph_module, *args))
            input_names = tuple((input.name for input in onnx_model.graph.input))
            output_names = tuple((output.name for output in onnx_model.graph.output))
            input_devices = _get_onnx_devices(args)
            if isinstance(prim_outputs, tuple):
                output_devices = _get_onnx_devices(prim_outputs)
            else:
                output_devices = _get_onnx_devices((prim_outputs,))
            execution_info_per_session = OrtExecutionInfoPerSession(session=onnx_session, input_names=input_names, input_value_infos=tuple((input for input in onnx_model.graph.input)), output_names=output_names, output_value_infos=tuple((output for output in onnx_model.graph.output)), input_devices=input_devices, output_devices=output_devices, example_outputs=prim_outputs)
            self._all_ort_execution_info.cache_session_execution_info(graph_module, execution_info_per_session)
        self.execution_count += 1
        is_single_tensor_output = isinstance(prim_outputs, torch.Tensor)
        normalized_prim_outputs = (prim_outputs,) if is_single_tensor_output else prim_outputs
        assert isinstance(normalized_prim_outputs, tuple)
        assert all((isinstance(elem, torch.Tensor) for elem in normalized_prim_outputs))
        _nvtx_range_push('run_onnx_session_with_ortvaluevector')
        onnx_outputs = self.run(onnx_session, input_names, args, input_devices, output_names, normalized_prim_outputs, output_devices, self._options.preallocate_output)
        _nvtx_range_pop()
        if self._assert_allclose_to_baseline:
            baseline_outputs = torch._prims.executor.execute(graph_module, *args, executor='aten')
            normalized_baseline_ouptuts = (baseline_outputs,) if is_single_tensor_output else baseline_outputs
            for (onnx_output, baseline_output) in zip(onnx_outputs, normalized_baseline_ouptuts):
                torch.testing.assert_close(onnx_output, baseline_output)
        return onnx_outputs[0] if is_single_tensor_output else onnx_outputs

    def compile(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        if False:
            while True:
                i = 10
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        if graph_module in self._partitioner_cache:
            partitioned_prim_graph_module = self._partitioner_cache[graph_module]
        else:
            prim_graph_module = graph_module
            _replace_to_copy_with_to(prim_graph_module)
            partitioner = CapabilityBasedPartitioner(prim_graph_module, self._supported_ops, allows_single_node_partition=True)
            partitioned_prim_graph_module = partitioner.partition_and_fuse()
            self._partitioner_cache[graph_module] = partitioned_prim_graph_module
            for node in partitioned_prim_graph_module.graph.nodes:
                if node.op == 'call_module' and 'fused_' in node.name:
                    fused_module = getattr(partitioned_prim_graph_module, node.name)
                    fused_module._wrapped_call = self._ort_acclerated_call
        return partitioned_prim_graph_module

    def __call__(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        if False:
            return 10
        "If ``OrtBackendOptions.use_aot_autograd`` is ``True``, the `auto_autograd` compiler\n        will be invoked, wrapping this ``OrtBackend`` instance's ``compile`` method. Otherwise,\n        the ``compile`` method is invoked directly."
        if self._options.use_aot_autograd:
            from functorch.compile import min_cut_rematerialization_partition
            from torch._dynamo.backends.common import aot_autograd
            return aot_autograd(fw_compiler=self.compile, partition_fn=min_cut_rematerialization_partition, decompositions=self._resolved_onnx_exporter_options.decomposition_table)(graph_module, args)
        return self.compile(graph_module, args)
    __instance_cache_max_count: Final = 8
    __instance_cache: Final[List['OrtBackend']] = []

    @staticmethod
    def get_cached_instance_for_options(options: Optional[Union[OrtBackendOptions, Mapping[str, Any]]]=None) -> 'OrtBackend':
        if False:
            return 10
        'Returns a possibly cached instance of an ``OrtBackend``. If an existing\n        backend was created previously through this function with the same options,\n        it will be returned. Otherwise a new backend will be created, cached, and\n        returned.\n\n        Note: if ``options`` sets ``ort_session_options``, a new ``OrtBackend``\n        will always be returned, since ``onnxruntime.SessionOptions`` cannot\n        participate in caching.'

        def reusable(a: OrtBackendOptions, b: OrtBackendOptions):
            if False:
                for i in range(10):
                    print('nop')
            if a.preferred_execution_providers != b.preferred_execution_providers or a.infer_execution_providers != b.infer_execution_providers or a.default_execution_providers != b.default_execution_providers or (a.preallocate_output != b.preallocate_output) or (a.use_aot_autograd != b.use_aot_autograd):
                return False
            if a.ort_session_options is not None or b.ort_session_options is not None:
                return False
            if a.export_options is b.export_options:
                return True
            if a.export_options is not None and b.export_options is not None:
                return a.export_options.dynamic_shapes == b.export_options.dynamic_shapes and a.export_options.op_level_debug == b.export_options.op_level_debug and (a.export_options.diagnostic_options == b.export_options.diagnostic_options) and (a.export_options.onnx_registry is b.export_options.onnx_registry) and (a.export_options.fake_context is b.export_options.fake_context)
            return False
        if not isinstance(options, OrtBackendOptions):
            options = OrtBackendOptions(**options or {})
        backend = next((b for b in OrtBackend.__instance_cache if reusable(b._options, options)), None)
        if backend is None:
            assert len(OrtBackend.__instance_cache) < OrtBackend.__instance_cache_max_count, f'No more than {OrtBackend.__instance_cache_max_count} instances of {OrtBackend} allowed. Please instantiate `{OrtBackend}` explicitly to pass to `torch.compile`. See https://github.com/pytorch/pytorch/pull/107973#discussion_r1306144795 for discussion.'
            OrtBackend.__instance_cache.append((backend := OrtBackend(options)))
        return backend

    @staticmethod
    def clear_cached_instances():
        if False:
            i = 10
            return i + 15
        OrtBackend.__instance_cache.clear()

    @staticmethod
    def get_cached_instances():
        if False:
            print('Hello World!')
        return tuple(OrtBackend.__instance_cache)

@compatibility(is_backward_compatible=False)
def torch_compile_backend(graph_module: torch.fx.GraphModule, args, *, options: Optional[Union[OrtBackendOptions, Mapping[str, Any]]]=None):
    if False:
        while True:
            i = 10
    return OrtBackend.get_cached_instance_for_options(options)(graph_module, args)