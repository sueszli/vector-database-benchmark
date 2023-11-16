import contextlib
import dataclasses
import functools
import logging
import os
import sys
import time
import warnings
from itertools import count
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union
from unittest import mock
from functorch.compile import min_cut_rematerialization_partition
import torch._functorch.config as functorch_config
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo import compiled_autograd, logging as dynamo_logging, utils as dynamo_utils
from torch._dynamo.utils import detect_fake_mode, lazy_format_graph_code
from torch._functorch.aot_autograd import aot_export_module, make_boxed_func
from torch._inductor.codecache import code_hash, CompiledFxGraph, FxGraphCache
from torch._inductor.debug import save_args_for_compile_fx_inner
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from .._dynamo.backends.common import aot_autograd
from ..fx.graph import _PyTreeCodeGen
from . import config, metrics
from .debug import DebugContext
from .decomposition import select_decomp_table
from .fx_passes.joint_graph import joint_graph_passes
from .fx_passes.post_grad import post_grad_passes, view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes
from .graph import GraphLowering
from .ir import ExternKernelNode
from .pattern_matcher import clone_graph
from .utils import get_dtype_size, has_incompatible_cudagraph_ops
from .virtualized import V
if config.is_fbcode():
    from torch._inductor.fb.utils import time_and_log
else:

    def time_and_log(attr: str):
        if False:
            print('Hello World!')
        return dynamo_utils.identity
log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, 'perf_hints')
post_grad_graphs_log = torch._logging.getArtifactLogger(__name__, 'post_grad_graphs')
ALIGNMENT = 16

@dataclasses.dataclass
class BoxedBool:
    value: bool

    def __bool__(self):
        if False:
            print('Hello World!')
        return self.value

    @staticmethod
    def disable(obj):
        if False:
            return 10
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False

@dataclasses.dataclass
class BoxedDeviceIndex:
    value: Optional[int]

    def set(self, device_idx):
        if False:
            print('Hello World!')
        assert device_idx is None or isinstance(device_idx, int)
        self.value = device_idx

def get_expanded_dims(t):
    if False:
        return 10
    if not isinstance(t, torch.Tensor):
        return None
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]

def index_expanded_dims(t: torch.Tensor, expanded_dims: List[int]) -> torch.Tensor:
    if False:
        print('Hello World!')
    for expanded_dim in expanded_dims:
        t = torch.ops.aten.slice(t, expanded_dim, 0, 1)
    return t

def complex_memory_overlap(t: torch.Tensor) -> bool:
    if False:
        for i in range(10):
            print('nop')
    t = index_expanded_dims(t, get_expanded_dims(t))
    if torch._debug_has_internal_overlap(t) != 0:
        strides = t.stride()
        sizes = t.shape
        indices = list(range(len(strides)))
        indices = [x for (_, x) in sorted(zip(strides, indices))]
        for i in range(len(strides)):
            prev_stride = 1 if i == 0 else strides[indices[i - 1]]
            prev_size = 1 if i == 0 else sizes[indices[i - 1]]
            if strides[indices[i]] < prev_stride * prev_size:
                return True
    return False

@functools.lru_cache(None)
def _step_logger():
    if False:
        i = 10
        return i + 15
    return dynamo_logging.get_step_logger(log)

@functools.lru_cache(None)
def _warn_tf32_disabled():
    if False:
        i = 10
        return i + 15
    if torch.cuda.is_available() and (not torch.backends.cuda.matmul.allow_tf32) and (torch.cuda.get_device_capability() >= (8, 0)):
        warnings.warn("TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.")

def _unlift_graph(mod, gm, graph_signature):
    if False:
        print('Hello World!')
    state_dict = {}
    for (name, param) in mod.named_parameters(remove_duplicate=False):
        state_dict[name] = param
    for (name, param) in mod.named_buffers(remove_duplicate=False):
        state_dict[name] = param
    from torch._export.exported_program import _construct_inp_pos_to_param_buffer_name, _unlift
    inp_pos_to_param_buffer_name = _construct_inp_pos_to_param_buffer_name(gm, graph_signature, state_dict, {})
    unlifted_gm = _unlift(gm, inp_pos_to_param_buffer_name, pytree.LeafSpec(), None, state_dict, {}, graph_signature.buffers_to_mutate)
    return unlifted_gm

def is_tf32_warning_applicable(gm: torch.fx.GraphModule):
    if False:
        while True:
            i = 10
    aten = torch.ops.aten
    tf32_ops = {aten.mm.default, aten.addmm.default, aten.bmm.default, aten.baddbmm.default}
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target in tf32_ops and isinstance(node.meta.get('val', None), torch.Tensor) and (node.meta['val'].dtype == torch.float32) and (node.meta['val'].device.type == 'cuda'):
            return True
    return False

@DebugContext.wrap
def count_bytes_inner(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], num_fixed: int=0, **kwargs):
    if False:
        i = 10
        return i + 15
    shape_env = _shape_env_from_inputs(example_inputs)
    fake_mode = fake_tensor_prop(gm, example_inputs)
    with V.set_fake_mode(fake_mode):
        post_grad_passes(gm, False)
    graph = GraphLowering(gm, shape_env=shape_env, num_static_inputs=num_fixed)
    with V.set_graph_handler(graph), V.set_real_inputs(example_inputs):
        graph.run(*example_inputs)
        (num_bytes, nodes_num_elem, node_runtimes) = graph.count_bytes()
        metrics.num_bytes_accessed += num_bytes
        metrics.nodes_num_elem += nodes_num_elem
        metrics.node_runtimes += node_runtimes
    return make_boxed_func(gm.forward)

def inner_compile_with_cpp_wrapper(inner_compile: Callable[..., Any]):
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(inner_compile)
    def wrapper(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Compile into cpp wrapper:\n        For CPU, this is currently done in one pass.\n        For GPU, this is done in two passes: JIT-compile the model with python wrapper code\n        and run it to generate autotuned kernel binaries in the first pass; and then generate\n        cpp wrapper code and compile it to a dynamic library in the second pass.\n        '
        devices = {t.device.type for t in gm.parameters()} | {t.device.type for t in gm.buffers()} | {t.device.type for t in example_inputs if isinstance(t, torch.Tensor)}
        if 'cuda' not in devices:
            kwargs_patched = {**kwargs, 'cpp_wrapper': True}
            return inner_compile(gm, example_inputs, **kwargs_patched)
        else:
            with config.patch({'triton.store_cubin': True}):
                kwargs_patched = {**kwargs, 'cpp_wrapper': False}
                compiled = inner_compile(clone_graph(gm), example_inputs, **kwargs_patched)

                def materialize(x):
                    if False:
                        i = 10
                        return i + 15
                    if isinstance(x, (torch.SymInt, torch.SymFloat)):
                        return x.node.hint
                    else:
                        assert not isinstance(x, FakeTensor)
                        return x
                if (tracing_context := torch._guards.TracingContext.try_get()):
                    if tracing_context.output_strides:
                        tracing_context.output_strides.clear()
                    params_flat = [param for param in tracing_context.params_flat if param is not None]
                    real_inputs = [materialize(x) for x in params_flat + V.real_inputs]
                else:
                    real_inputs = [materialize(x) for x in V.real_inputs]
                with torch.utils._python_dispatch._disable_current_modes():
                    compiled(real_inputs)
                del real_inputs
                kwargs_patched = {**kwargs, 'cpp_wrapper': True}
                return inner_compile(gm, example_inputs, **kwargs_patched)
    return wrapper

def fake_tensor_prop(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], force_allow_non_fake_inputs: bool=False):
    if False:
        while True:
            i = 10
    '\n    If we can not detect fake mode from the context of inputs, create one.\n\n    The created fake mode will be returned.\n    '
    fake_mode = detect_fake_mode(example_inputs)
    if not fake_mode:
        fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
    else:
        ctx = contextlib.nullcontext() if not force_allow_non_fake_inputs else mock.patch.object(fake_mode, 'allow_non_fake_inputs', True)
        with ctx:
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*example_inputs)
    return fake_mode

@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
@time_and_log(attr='compilation time (in seconds)')
def compile_fx_inner(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], cudagraphs: Optional[BoxedBool]=None, num_fixed: int=0, is_backward: bool=False, graph_id: Optional[int]=None, cpp_wrapper: bool=False, aot_mode: bool=False, is_inference: bool=False, boxed_forward_device_index: Optional[BoxedDeviceIndex]=None, user_visible_outputs: FrozenSet[str]=frozenset(), layout_opt: Optional[bool]=None, extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]]=None) -> Union[CompiledFxGraph, str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Inductor API that compiles a single graph.\n\n    If you change the argument list for this function, make sure you\n    also update the call to save_args_for_compile_fx_inner below accordingly.\n    '
    if dynamo_utils.count_calls(gm.graph) == 0 and (not aot_mode):
        return make_boxed_func(gm.forward)
    assert isinstance(next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)), f'inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}'
    if config.save_args:
        save_args_for_compile_fx_inner(gm, example_inputs, cudagraphs=cudagraphs, num_fixed=num_fixed, is_backward=is_backward, graph_id=graph_id, cpp_wrapper=cpp_wrapper, aot_mode=aot_mode, is_inference=is_inference, boxed_forward_device_index=boxed_forward_device_index, user_visible_outputs=user_visible_outputs, layout_opt=layout_opt)
    if cudagraphs is None:
        cudagraphs = BoxedBool(config.triton.cudagraphs)
    graph_kwargs = {'cudagraphs': cudagraphs, 'num_fixed': num_fixed, 'is_backward': is_backward, 'graph_id': graph_id, 'cpp_wrapper': cpp_wrapper, 'aot_mode': aot_mode, 'is_inference': is_inference, 'user_visible_outputs': user_visible_outputs, 'layout_opt': layout_opt, 'extern_node_serializer': extern_node_serializer}
    start = time.time()
    if config.fx_graph_cache and (not aot_mode):
        compiled_graph = FxGraphCache.load(fx_codegen_and_compile, gm, example_inputs, graph_kwargs)
    else:
        compiled_graph = fx_codegen_and_compile(gm, example_inputs, **graph_kwargs)
    log.debug('FX codegen and compilation took %.3fs', time.time() - start)
    context = torch._guards.TracingContext.try_get()
    if context is not None and context.output_strides is not None:
        assert len(context.output_strides) == 0
        context.output_strides.extend(compiled_graph.output_strides)
    if aot_mode:
        return compiled_graph
    if cudagraphs:
        output = list(gm.graph.nodes)[-1]
        assert len(output.args) == 1
        stack_traces = [arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None for arg in output.args[0]]
        complex_memory_overlap_inputs = any((complex_memory_overlap(t) for t in example_inputs if isinstance(t, torch.Tensor)))
        if config.triton.cudagraph_trees:
            has_mutation = not all((idx < num_fixed for idx in compiled_graph.mutated_input_idxs))
        else:
            has_mutation = len(compiled_graph.mutated_inputs) != 0
        cudagraph_tests = [(set(compiled_graph.device_types) == {'cuda'}, 'non-cuda device in graph'), (not has_mutation, 'mutated inputs'), (not has_incompatible_cudagraph_ops(gm), 'incompatible ops'), (not complex_memory_overlap_inputs, 'complex memory overlap'), (all((isinstance(t, (torch.Tensor, torch.SymInt)) for t in example_inputs)), 'non-Tensor inputs'), (len(compiled_graph.device_idxs) == 1 or not config.triton.cudagraph_trees, 'multiple device indices without cudagraph_trees')]
        cudagraph_fail_reasons = [s for (b, s) in cudagraph_tests if not b]
        if not cudagraph_fail_reasons:
            if not config.triton.cudagraph_trees:
                for t in example_inputs:
                    if isinstance(t, torch.SymInt):
                        int(t)
            if boxed_forward_device_index is not None and (not is_inference) and (not is_backward):
                boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))
            compiled_graph.current_callable = cudagraphify(compiled_graph.get_current_callable(), example_inputs, static_input_idxs=range(num_fixed), device_index=next(iter(compiled_graph.device_idxs)), stack_traces=stack_traces, is_backward=is_backward, is_inference=is_inference, constants=tuple(compiled_graph.constants.values()))
        else:
            BoxedBool.disable(cudagraphs)
            if is_backward and config.triton.cudagraph_trees:
                assert boxed_forward_device_index is not None
                assert boxed_forward_device_index.value is not None
                compiled_graph_callable = compiled_graph.get_current_callable()
                manager = torch._inductor.cudagraph_trees.get_manager(boxed_forward_device_index.value, create_if_none_exists=False)
                assert manager is not None

                def compiled_artifact(new_inputs):
                    if False:
                        print('Hello World!')
                    manager.set_to_running_backward()
                    return compiled_graph_callable(new_inputs)
                compiled_graph.current_callable = compiled_artifact
            if 'cuda' in compiled_graph.device_types:
                perf_hint_log.warning('skipping cudagraphs due to %s', cudagraph_fail_reasons)
    if not cudagraphs:
        new_callable = align_inputs(compiled_graph.get_current_callable(), example_inputs, range(num_fixed))
        if new_callable is not compiled_graph.get_current_callable():
            compiled_graph.current_callable = new_callable
    _step_logger()(logging.INFO, f"torchinductor done compiling {('BACKWARDS' if is_backward else 'FORWARDS')} graph {graph_id}")
    compiled_graph._boxed_call = True
    return compiled_graph

def fx_codegen_and_compile(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], cudagraphs: Optional[BoxedBool]=None, num_fixed: int=0, is_backward: bool=False, graph_id: Optional[int]=None, cpp_wrapper: bool=False, aot_mode: bool=False, is_inference: bool=False, user_visible_outputs: FrozenSet[str]=frozenset(), layout_opt: Optional[bool]=None, extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]]=None) -> Union[CompiledFxGraph, str]:
    if False:
        while True:
            i = 10
    if is_tf32_warning_applicable(gm):
        _warn_tf32_disabled()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
    _step_logger()(logging.INFO, f"torchinductor compiling {('BACKWARDS' if is_backward else 'FORWARDS')} graph {graph_id}")
    V.debug.fx_graph(gm, example_inputs)
    shape_env = _shape_env_from_inputs(example_inputs)
    view_to_reshape(gm)
    with torch.no_grad():
        fake_mode = fake_tensor_prop(gm, example_inputs)
    with V.set_fake_mode(fake_mode):
        post_grad_passes(gm, is_inference=is_inference)
        V.debug.fx_graph_transformed(gm, example_inputs)
        post_grad_graphs_log.info('%s', lazy_format_graph_code('AFTER POST GRAD', gm))
    with V.set_fake_mode(fake_mode):
        graph = GraphLowering(gm, shape_env=shape_env, num_static_inputs=num_fixed, graph_id=graph_id, cpp_wrapper=cpp_wrapper, aot_mode=aot_mode, user_visible_outputs=user_visible_outputs, extern_node_serializer=extern_node_serializer, is_inference=is_inference)
        with V.set_graph_handler(graph):
            graph.run(*example_inputs)
            output_strides: List[Optional[Tuple[int, ...]]] = []
            if graph.graph_outputs is not None:
                for out in graph.graph_outputs:
                    if hasattr(out, 'layout'):
                        output_strides.append(tuple((V.graph.sizevars.size_hint(s) for s in out.layout.stride)))
                    else:
                        output_strides.append(None)
            compiled_fn = graph.compile_to_fn()
            if V.aot_compilation is True:
                return compiled_fn
            if graph.disable_cudagraphs:
                BoxedBool.disable(cudagraphs)
            compiled_graph = CompiledFxGraph(compiled_fn, graph, output_strides)
    return compiled_graph

def clone_preserve_strides(x: torch.Tensor):
    if False:
        print('Hello World!')
    needed_size = sum(((shape - 1) * stride for (shape, stride) in zip(x.size(), x.stride()))) + 1
    buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
    return torch.as_strided(buffer, x.size(), x.stride())

def copy_misaligned_inputs(new_inputs: List[torch.Tensor], check_inputs_idxs: Sequence[int]) -> None:
    if False:
        return 10
    for i in check_inputs_idxs:
        if new_inputs[i].data_ptr() % ALIGNMENT:
            new_inputs[i] = clone_preserve_strides(new_inputs[i])

def get_input_idxs_to_check(inputs: Union[List[torch.Tensor], Sequence[int]], static_input_idxs: Sequence[int]) -> Sequence[int]:
    if False:
        return 10

    def is_aligned(storage_offset, dtype):
        if False:
            print('Hello World!')
        return storage_offset * get_dtype_size(dtype) % ALIGNMENT == 0
    ids_to_check = []
    for (i, input) in enumerate(inputs):
        if isinstance(input, torch.Tensor) and (i not in static_input_idxs or not is_aligned(input.storage_offset(), input.dtype)) and (input.device.type == 'cuda'):
            ids_to_check.append(i)
    return ids_to_check

def align_inputs_from_check_idxs(model: Callable[[List[torch.Tensor]], Any], inputs_to_check: Sequence[int]):
    if False:
        for i in range(10):
            print('nop')
    if len(inputs_to_check) == 0:
        return model

    def run(new_inputs):
        if False:
            for i in range(10):
                print('nop')
        copy_misaligned_inputs(new_inputs, inputs_to_check)
        return model(new_inputs)
    return run

def align_inputs(model: Callable[[List[torch.Tensor]], Any], inputs: List[torch.Tensor], static_input_idxs: Sequence[int]=()):
    if False:
        while True:
            i = 10
    inputs_to_check = get_input_idxs_to_check(inputs, static_input_idxs)
    return align_inputs_from_check_idxs(model, inputs_to_check)

@dynamo_utils.dynamo_timed
def cudagraphify(model: torch.fx.GraphModule, inputs: List[torch.Tensor], static_input_idxs: Sequence[int]=(), *, device_index: int, stack_traces: List[Optional[str]], is_backward: bool, is_inference: bool, constants: Tuple[torch.Tensor, ...]=()):
    if False:
        i = 10
        return i + 15
    from torch._inductor.cudagraph_trees import cudagraphify_impl as new_cudagraphify_impl
    cudagraphify_fn: Callable[..., Any]
    if config.triton.cudagraph_trees:
        cudagraphify_fn = functools.partial(new_cudagraphify_impl, device_index=device_index, stack_traces=stack_traces, is_backward=is_backward, is_inference=is_inference, constants=constants)
    else:
        cudagraphify_fn = cudagraphify_impl
    if not any((isinstance(inp, FakeTensor) for inp in inputs)):
        return cudagraphify_fn(model, inputs, static_input_idxs)
    compiled_fn = None

    def run(new_inputs):
        if False:
            i = 10
            return i + 15
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
        return compiled_fn(new_inputs)
    return run

def remove_unaligned_input_idxs(inputs: Union[List[torch.Tensor], Sequence[int]], static_input_idxs: Sequence[int]):
    if False:
        while True:
            i = 10
    "\n    We require all inputs to be aligned, so introduce a copy for any\n    that aren't.\n    "
    aligned_static_input_idxs = []
    for (idx, input) in zip(static_input_idxs, inputs):
        if isinstance(input, torch.Tensor) and input.data_ptr() % ALIGNMENT == 0:
            aligned_static_input_idxs.append(idx)
    if len(aligned_static_input_idxs) != len(static_input_idxs):
        return aligned_static_input_idxs
    return static_input_idxs

def static_input(x: torch.Tensor):
    if False:
        print('Hello World!')
    '\n    Copy and input while preserving strides\n    '
    needed_size = sum(((shape - 1) * stride for (shape, stride) in zip(x.size(), x.stride()))) + 1
    buffer = torch.empty(needed_size, dtype=x.dtype, device=x.device)
    return torch.as_strided(buffer, x.size(), x.stride())

def index_expanded_dims_and_copy_(dst: torch.Tensor, src: torch.Tensor, expanded_dims: List[int]):
    if False:
        print('Hello World!')
    'Index into expanded dimensions of both dst and src then copy_'
    dst = index_expanded_dims(dst, expanded_dims)
    src = index_expanded_dims(src, expanded_dims)
    dst.copy_(src)

def cudagraphify_impl(model: torch.fx.GraphModule, inputs: List[torch.Tensor], static_input_idxs: Sequence[int]=()):
    if False:
        for i in range(10):
            print('nop')
    '\n    Assumes inputs[static_input_idxs[i]] are always the same memory address\n    '
    check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
    static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
    copy_misaligned_inputs(inputs, check_input_idxs)
    assert isinstance(inputs, list)
    inps_expanded_dims = [get_expanded_dims(x) if idx not in static_input_idxs else [] for (idx, x) in enumerate(inputs)]
    static_inputs = [x if not isinstance(x, torch.Tensor) else static_input(x) if idx not in static_input_idxs else x.detach() for (idx, x) in enumerate(inputs)]
    for (idx, (x, expanded_dims)) in enumerate(zip(inputs, inps_expanded_dims)):
        if isinstance(x, torch.Tensor) and idx not in static_input_idxs:
            index_expanded_dims_and_copy_(static_inputs[idx], x, expanded_dims)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)
    if config.size_asserts:

        def run(new_inputs):
            if False:
                return 10
            assert len(static_inputs) == len(new_inputs)
            for (idx, (dst, src, expanded_dims)) in enumerate(zip(static_inputs, new_inputs, inps_expanded_dims)):
                if not isinstance(dst, torch.Tensor):
                    pass
                elif idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    index_expanded_dims_and_copy_(dst, src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            return static_outputs
    else:
        copy_indices = [idx for idx in range(len(static_inputs)) if idx not in static_input_idxs]

        def run(new_inputs):
            if False:
                i = 10
                return i + 15
            for idx in copy_indices:
                expanded_dims = inps_expanded_dims[idx]
                index_expanded_dims_and_copy_(static_inputs[idx], new_inputs[idx], expanded_dims)
            new_inputs.clear()
            graph.replay()
            return static_outputs
    return align_inputs_from_check_idxs(run, check_input_idxs)

def count_tangents(fx_g: torch.fx.GraphModule):
    if False:
        for i in range(10):
            print('nop')
    '\n    Infers which inputs are static for a backwards graph\n    '

    def is_saved_tensor(x):
        if False:
            i = 10
            return i + 15
        return 'tangents' not in x.name and 'bwd_seed' not in x.name and ('bwd_base_offset' not in x.name)
    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == 'placeholder':
            if is_saved_tensor(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1
    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)

def compile_fx_aot(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor], inner_compile: Callable[..., Any]=compile_fx_inner, config_patches: Optional[Dict[str, Any]]=None):
    if False:
        return 10
    config_patches: Dict[str, Any] = {'cpp_wrapper': True} if config_patches is None else {**config_patches, 'cpp_wrapper': True}
    if 'aot_inductor.output_path' not in config_patches and (not config.aot_inductor.output_path):
        config_patches = {**config_patches, 'aot_inductor.output_path': code_hash(model_.code)}
    extern_node_serializer = config_patches.pop('extern_node_serializer', None)
    with V.set_aot_compilation(True):
        compiled_lib_path = compile_fx(model_, example_inputs_, inner_compile=functools.partial(inner_compile, aot_mode=True, extern_node_serializer=extern_node_serializer), config_patches=config_patches)
        assert os.path.exists(compiled_lib_path), f'AOTInductor compiled library does not exist at {compiled_lib_path}'
        return compiled_lib_path
_graph_counter = count(0)

def fw_compiler_freezing(aot_autograd_model: torch.fx.GraphModule, aot_example_inputs: List[torch.Tensor], dynamo_model: torch.fx.GraphModule, num_example_inputs: int, inner_compile: Callable[..., Any], cudagraphs: BoxedBool, graph_id: int, forward_device: BoxedDeviceIndex):
    if False:
        for i in range(10):
            print('nop')
    from torch._inductor.freezing import convert_conv_weights_to_channels_last, freeze
    joint_graph_passes(aot_autograd_model)
    layout_opt = GraphLowering.decide_layout_opt(aot_autograd_model)
    if layout_opt:
        fake_tensor_prop(aot_autograd_model, aot_example_inputs, True)
        convert_conv_weights_to_channels_last(aot_autograd_model)
    (opt_model, preserved_arg_indices) = freeze(dynamo_model, aot_autograd_model, aot_example_inputs)
    aot_example_inputs = [aot_example_inputs[ind] for ind in preserved_arg_indices]
    num_fixed = len(preserved_arg_indices) - num_example_inputs
    fake_mode = detect_fake_mode(aot_example_inputs)
    (*_, model_outputs_node) = opt_model.graph.nodes
    model_outputs = model_outputs_node.args[0]
    user_visible_outputs = [n.name for n in model_outputs if isinstance(n, torch.fx.Node)]
    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context is not None:
        params_flat = tracing_context.params_flat
        assert params_flat is not None
        for i in range(len(params_flat)):
            if i not in preserved_arg_indices:
                params_flat[i] = None
    with mock.patch.object(fake_mode, 'allow_non_fake_inputs', True):
        optimized_function = inner_compile(opt_model, aot_example_inputs, num_fixed=num_fixed, cudagraphs=cudagraphs, graph_id=graph_id, is_inference=True, boxed_forward_device_index=forward_device, layout_opt=layout_opt, user_visible_outputs=user_visible_outputs)
    if V.aot_compilation is True:
        return optimized_function

    def wrapper(args):
        if False:
            print('Hello World!')
        args_new = [args[i] for i in preserved_arg_indices]
        args.clear()
        return optimized_function(args_new)
    wrapper._boxed_call = True
    return wrapper

def compile_fx(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor], inner_compile: Callable[..., Any]=compile_fx_inner, config_patches: Optional[Dict[str, Any]]=None, decompositions: Optional[Dict[OpOverload, Callable[..., Any]]]=None):
    if False:
        print('Hello World!')
    'Main entrypoint to a compile given FX graph'
    if config_patches:
        with config.patch(config_patches):
            return compile_fx(model_, example_inputs_, inner_compile=config.patch(config_patches)(inner_compile), decompositions=decompositions)
    if config.cpp_wrapper:
        with config.patch({'cpp_wrapper': False, 'triton.autotune_cublasLt': False, 'triton.cudagraphs': False}), V.set_real_inputs(example_inputs_):
            inputs_ = example_inputs_
            if isinstance(model_, torch.fx.GraphModule):
                fake_inputs = [node.meta.get('val') for node in model_.graph.nodes if node.op == 'placeholder']
                if all((v is not None for v in fake_inputs)):
                    for (idx, fi, i) in zip(count(), fake_inputs, inputs_):
                        if fi.device != i.device:
                            raise ValueError(f'Device mismatch between fake input and example input at position #{idx}: {fi.device} vs {i.device}. If the model was exported via torch.export(), make sure torch.export() and torch.aot_compile() run on the same device.')
                    inputs_ = fake_inputs
            return compile_fx(model_, inputs_, inner_compile=inner_compile_with_cpp_wrapper(inner_compile), decompositions=decompositions)
    recursive_compile_fx = functools.partial(compile_fx, inner_compile=inner_compile, decompositions=decompositions)
    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(model_, example_inputs_, recursive_compile_fx)
    if isinstance(model_, torch.fx.GraphModule):
        if isinstance(model_.graph._codegen, _PyTreeCodeGen):
            return handle_dynamo_export_graph(model_, example_inputs_, recursive_compile_fx)
        model_ = pre_grad_passes(model_, example_inputs_)
    if any((isinstance(x, (list, tuple, dict)) for x in example_inputs_)):
        return flatten_graph_inputs(model_, example_inputs_, recursive_compile_fx)
    assert not config._raise_error_for_testing
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(config.triton.cudagraphs)
    forward_device = BoxedDeviceIndex(None)
    graph_id = next(_graph_counter)
    decompositions = decompositions if decompositions is not None else select_decomp_table()

    @dynamo_utils.dynamo_timed
    def fw_compiler_base(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor], is_inference: bool):
        if False:
            while True:
                i = 10
        if is_inference:
            joint_graph_passes(model)
        num_rng_seed_offset_inputs = 2 if functorch_config.functionalize_rng_ops else 0
        fixed = len(example_inputs) - num_example_inputs - num_rng_seed_offset_inputs
        user_visible_outputs = set()
        if config.keep_output_stride:
            (*_, model_outputs_node) = model.graph.nodes
            assert model_outputs_node.op == 'output'
            model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
            num_model_outputs = len(model_outputs)
            context = torch._guards.TracingContext.try_get()
            if context is not None and context.fw_metadata and (not is_inference):
                original_output_start_index = context.fw_metadata.num_mutated_inp_runtime_indices
            else:
                original_output_start_index = 0
            if isinstance(model_, torch.fx.GraphModule):
                (*_, orig_model_outputs_node) = model_.graph.nodes
                assert orig_model_outputs_node.op == 'output'
                (orig_model_outputs, _) = pytree.tree_flatten(orig_model_outputs_node.args)
                num_orig_model_outputs = len(orig_model_outputs)
            else:
                num_orig_model_outputs = num_model_outputs
            assert num_orig_model_outputs <= num_model_outputs
            orig_output_end_idx = original_output_start_index + num_orig_model_outputs
            assert orig_output_end_idx <= num_model_outputs
            user_visible_outputs = {n.name for n in model_outputs[original_output_start_index:orig_output_end_idx] if isinstance(n, torch.fx.Node)}
        return inner_compile(model, example_inputs, num_fixed=fixed, cudagraphs=cudagraphs, graph_id=graph_id, is_inference=is_inference, boxed_forward_device_index=forward_device, user_visible_outputs=user_visible_outputs)
    fw_compiler = functools.partial(fw_compiler_base, is_inference=False)
    if config.freezing and (not torch.is_grad_enabled()):
        inference_compiler = functools.partial(fw_compiler_freezing, dynamo_model=model_, num_example_inputs=num_example_inputs, inner_compile=inner_compile, cudagraphs=cudagraphs, graph_id=graph_id, forward_device=forward_device)
    else:
        inference_compiler = functools.partial(fw_compiler_base, is_inference=True)

    def partition_fn(graph, joint_inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        joint_graph_passes(graph)
        return min_cut_rematerialization_partition(graph, joint_inputs, **kwargs, compiler='inductor')

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if False:
            return 10
        fixed = count_tangents(model)
        return inner_compile(model, example_inputs, num_fixed=fixed, cudagraphs=cudagraphs, is_backward=True, graph_id=graph_id, boxed_forward_device_index=forward_device)
    fake_mode = detect_fake_mode(example_inputs_) or torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
    tracing_context = torch._guards.TracingContext.try_get() or torch._guards.TracingContext(fake_mode)
    if V.aot_compilation is True:
        (gm, graph_signature) = aot_export_module(model_, example_inputs_, trace_joint=False, decompositions=decompositions)
        unlifted_gm = _unlift_graph(model_, gm, graph_signature)
        with V.set_fake_mode(fake_mode), compiled_autograd.disable():
            return inference_compiler(unlifted_gm, example_inputs_)
    with V.set_fake_mode(fake_mode), torch._guards.tracing(tracing_context), compiled_autograd.disable():
        return aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, inference_compiler=inference_compiler, decompositions=decompositions, partition_fn=partition_fn, keep_inference_input_mutations=True)(model_, example_inputs_)

def get_patched_config_dict(config_patches=None):
    if False:
        while True:
            i = 10
    with config.patch(config_patches):
        return config.get_config_copy()

def _shape_env_from_inputs(inputs: List[torch.Tensor]):
    if False:
        return 10
    shape_env = None
    fake_mode = detect_fake_mode(inputs)
    if fake_mode is not None:
        return fake_mode.shape_env
    for input in inputs:
        if isinstance(input, torch.SymInt):
            return input.node.shape_env
    return None

def output_node(gm: torch.fx.GraphModule):
    if False:
        while True:
            i = 10
    'Get the output node from an FX graph'
    last_node = next(iter(reversed(gm.graph.nodes)))
    assert last_node.op == 'output'
    return last_node

def graph_returns_tuple(gm: torch.fx.GraphModule):
    if False:
        i = 10
        return i + 15
    'True if a FX graph returns a tuple'
    if not isinstance(gm, torch.fx.GraphModule):
        return True
    (rv,) = output_node(gm).args
    if isinstance(rv, (list, tuple)):
        return True
    if isinstance(rv, torch.fx.node.Node) and hasattr(rv.target, '_schema') and (len(rv.target._schema.returns) > 1) and all((str(ret.type) == 'Tensor' for ret in rv.target._schema.returns)):
        return True
    return False

def make_graph_return_tuple(gm: torch.fx.GraphModule, inputs: List[torch.Tensor], compile_gm: Callable[..., Any]):
    if False:
        i = 10
        return i + 15
    '\n    Mutate gm so it returns a tuple.  This is only needed for graphs\n    not created by torchdynamo that return non-tuples.\n    '
    node = output_node(gm)
    (rv,) = node.args
    (rv, spec) = pytree.tree_flatten(rv)
    with gm.graph.inserting_before(node):
        gm.graph.output(rv)
    gm.graph.erase_node(node)
    assert graph_returns_tuple(gm)
    compiled_fn = compile_gm(gm, inputs)

    @functools.wraps(compiled_fn)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return pytree.tree_unflatten(compiled_fn(*args, **kwargs), spec)
    return wrapper

def flatten_graph_inputs(gm: torch.fx.GraphModule, inputs, compile_gm):
    if False:
        return 10
    '\n    Mutate inputs so that they are flat and wrap gm such that it\n    accepts those inputs.  This is only needed for graphs not created\n    by torchdynamo that take bumpy inputs.\n    '
    (inputs, spec) = pytree.tree_flatten(inputs)

    class GmWrapper(torch.nn.Module):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.gm = gm

        def forward(self, *args):
            if False:
                print('Hello World!')
            args: List[Any] = list(args)
            return self.gm(*pytree.tree_unflatten(args, spec))
    compiled_fn = compile_gm(GmWrapper(), inputs)

    @functools.wraps(compiled_fn)
    def wrapper(*args):
        if False:
            print('Hello World!')
        return compiled_fn(*pytree.arg_tree_leaves(*args))
    return wrapper

def handle_dynamo_export_graph(gm: torch.fx.GraphModule, inputs: List[torch.Tensor], compile_gm: Callable[..., Any]):
    if False:
        for i in range(10):
            print('nop')
    '\n    `torch._dynamo.export` embeds pytrees in the FX graph codegen object,\n    convert that to a normal FX graph so inductor can compile it.\n    '
    codegen = gm.graph._codegen
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.recompile()
    compiled_fn = compile_gm(gm, codegen.process_inputs(*inputs))

    @functools.wraps(compiled_fn)
    def wrapper(*args):
        if False:
            while True:
                i = 10
        return codegen.process_outputs(compiled_fn(*codegen.process_inputs(*args)))
    return wrapper