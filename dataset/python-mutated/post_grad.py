import functools
import itertools
import logging
import operator
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Union
from sympy import Expr
import torch
import torch._inductor as inductor
from torch._decomp import register_decomposition
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional
from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype
from torch.fx.immutable_collections import immutable_dict
from .. import config, inductor_prims, ir, pattern_matcher
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage
from ..lowering import lowerings as L
from ..pattern_matcher import _return_true, Arg, CallFunction, filter_nodes, get_arg_value, Ignored, init_once_fakemode, KeywordArg, ListOf, Match, MULTIPLE, PatternMatcherPass, register_graph_pattern, stable_topological_sort
from ..utils import decode_device, is_pointwise_use
from ..virtualized import V
from .group_batch_fusion import group_batch_fusion_post_grad_passes
log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
pass_patterns = [PatternMatcherPass(), PatternMatcherPass(), PatternMatcherPass()]
inference_patterns = PatternMatcherPass()

def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    if False:
        i = 10
        return i + 15
    '\n    Passes that run on after grad.  This is called once on the forwards\n    graph and once on the backwards graph.\n\n    The IR here has been normalized and functionalized.\n    '
    if config.dce:
        gm.graph.eliminate_dead_code()
    if is_inference and config.reorder_for_locality:
        reorder_for_locality(gm.graph)
    fake_tensor_updater = FakeTensorUpdater(gm.graph)
    if config.post_grad_custom_pre_pass is not None:
        config.post_grad_custom_pre_pass(gm.graph)
    if config.pattern_matcher:
        lazy_init()
        group_batch_fusion_post_grad_passes(gm.graph)
        remove_noop_ops(gm.graph)
        for patterns in pass_patterns:
            patterns.apply(gm.graph)
        if is_inference:
            inference_patterns.apply(gm.graph)
    if config.post_grad_custom_post_pass is not None:
        config.post_grad_custom_post_pass(gm.graph)
    stable_topological_sort(gm.graph)
    fake_tensor_updater.incremental_update()
    reinplace_inplaceable_ops(gm.graph)
    gm.recompile()
    gm.graph.lint()
    if config.is_fbcode():
        from torch._inductor.fb.utils import get_everpaste_url
        log.info('Print graph after recompile in post grad passes: %s', get_everpaste_url(str(gm.graph)))

@init_once_fakemode
def lazy_init():
    if False:
        while True:
            i = 10
    if torch._C._has_mkldnn:
        from .mkldnn_fusion import _mkldnn_fusion_init
        _mkldnn_fusion_init()

def reorder_for_locality(graph: torch.fx.Graph):
    if False:
        return 10

    def visit(other_node):
        if False:
            return 10
        if other_node.op == 'call_function' and other_node.target != operator.getitem and all((n in seen_nodes for n in other_node.users)):
            node.prepend(other_node)
    seen_nodes = set()
    first_copy = next((node for node in graph.nodes if node.op == 'call_function' and node.target == torch.ops.aten.copy_.default), None)
    past_mutating_epilogue = True if first_copy is None else False
    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        if not past_mutating_epilogue:
            past_mutating_epilogue = node is first_copy
            continue
        torch.fx.map_arg((node.args, node.kwargs), visit)

def register_lowering_pattern(pattern, extra_check=_return_true, pass_number=1):
    if False:
        print('Hello World!')
    '\n    Register an aten to inductor IR replacement pattern\n    '
    return pattern_matcher.register_lowering_pattern(pattern, extra_check, pass_dict=pass_patterns[pass_number])

@register_lowering_pattern(CallFunction(aten.add, CallFunction(aten.mm, Arg(), Arg()), CallFunction(aten.mm, Arg(), Arg())))
def mm_plus_mm(match: Match, mat1, mat2, mat3, mat4):
    if False:
        i = 10
        return i + 15
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)

def cuda_and_enabled_mixed_mm(match):
    if False:
        for i in range(10):
            print('nop')
    return (config.use_mixed_mm or config.force_mixed_mm) and getattr(match.kwargs['mat1'].meta.get('val'), 'is_cuda', False)

def cuda_and_enabled_mixed_mm_and_not_int8(match):
    if False:
        return 10
    return cuda_and_enabled_mixed_mm(match) and getattr(match.kwargs['mat1'].meta.get('val'), 'is_cuda', False) and (getattr(match.kwargs['mat2'].meta.get('val'), 'dtype', torch.int8) != torch.int8)
'\n    this is intended to be used to unpack a [K,N] int4 tensor from a [K/2, N] uint4x2 tensor\n    (where the int4 and uint4x2 are represented with int8 and uint8 respectively)\n    where every other row of the int4 is packed with the row above it as:\n    uint4x2[k,n] = (8+int4[2*k,n])+(8+int4[2*k+1,n])<<4\n\n    unpack formulas:\n    int4[2*k,n]=(uint4x2[k,n] & 0xF) - 8\n    int4[2*k+1,n]=(uint4x2[k,n] >> 4) - 8\n\n    thus matching on unpack formula:\n    torch.mm(mat1, torch.cat((mat2 & 0xF, mat2>>4),1).reshape(mat2_mm_shape).to(mat2_dtype).sub(8))\n\n    note: although the unpack formula in pytorch and the triton kernel is designed for a uint8 mat2, the behavior\n    of the kernel matches the pytorch formula for all dtypes except torch.int8\n    where the bitwise numerics in triton do not match those in pytorch.\n'

@register_lowering_pattern(CallFunction(aten.mm.default, KeywordArg('mat1'), CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, CallFunction(aten.reshape.default, CallFunction(aten.cat.default, ListOf(CallFunction(aten.bitwise_and.Scalar, KeywordArg('mat2'), 15), CallFunction(aten.__rshift__.Scalar, KeywordArg('mat2'), 4)), 1), KeywordArg('mat2_mm_shape')), KeywordArg('mat2_dtype')), 8)), extra_check=cuda_and_enabled_mixed_mm_and_not_int8)
def uint4x2_mixed_mm(match: Match, mat1, mat2, mat2_mm_shape, mat2_dtype):
    if False:
        i = 10
        return i + 15
    return inductor.kernel.unpack_mixed_mm.tuned_uint4x2_mixed_mm(mat1, mat2, mat2_mm_shape, mat2_dtype)
'\n    torch.mm(mat1, mat2.to(mat2_dtype))\n'

@register_lowering_pattern(CallFunction(aten.mm, KeywordArg('mat1'), CallFunction(prims.convert_element_type.default, KeywordArg('mat2'), KeywordArg('mat2_dtype'))), extra_check=cuda_and_enabled_mixed_mm)
def mixed_mm(match: Match, mat1, mat2, mat2_dtype):
    if False:
        for i in range(10):
            print('nop')
    return inductor.kernel.mm.tuned_mixed_mm(mat1, mat2, mat2_dtype)

@register_graph_pattern(CallFunction(aten.cumsum.default, CallFunction(torch.ops.aten.full.default, KeywordArg('shape'), KeywordArg('fill_value'), dtype=KeywordArg('dtype'), layout=Ignored(), device=KeywordArg('device'), pin_memory=False, _users=MULTIPLE), KeywordArg('dim'), _users=MULTIPLE), pass_dict=pass_patterns[1])
def pointless_cumsum_replacement(match: Match, shape, fill_value, device, dtype, dim):
    if False:
        i = 10
        return i + 15
    'Based on a pattern in OPTForCausalLM'
    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        dtype = torch.int64

    def repl(*shape):
        if False:
            return 10
        dim_size = shape[dim]
        idx = torch.arange(1, dim_size + 1, device=device, dtype=dtype)
        inter_shape = [1] * len(shape)
        inter_shape[dim] = dim_size
        return (idx * fill_value).view(inter_shape).expand(shape)
    match.nodes = [match.output_node()]
    with V.fake_mode:
        match.replace_by_example(repl, list(shape))

def shape_of_mm(a, b):
    if False:
        print('Hello World!')
    (m, _) = a.get_size()
    (_, n) = b.get_size()
    return [m, n]

@register_lowering_pattern(CallFunction(aten.cat, ListOf(CallFunction(aten.mm, Arg(), Arg())), Arg()))
def cat_mm(match, inputs, dim):
    if False:
        for i in range(10):
            print('nop')
    return cat_tuned_op(match, inputs, dim, op=L[aten.mm], shape_of=shape_of_mm)

@register_lowering_pattern(CallFunction(aten.cat, ListOf(CallFunction(aten.addmm, Arg(), Arg(), Arg())), Arg()))
def cat_addmm(match, inputs, dim):
    if False:
        return 10

    def shape_of(bias, a, b):
        if False:
            while True:
                i = 10
        (m, _) = a.get_size()
        (_, n) = b.get_size()
        return [m, n]
    return cat_tuned_op(match, inputs, dim, op=L[aten.addmm], shape_of=shape_of)

def cat_tuned_op(match, inputs, dim, *, op, shape_of):
    if False:
        while True:
            i = 10
    "\n    Memory planning to remove cat. We can't use the stock memory\n    planner since autotuning matmuls needs to know the output layout.\n    "
    if len(inputs) == 1:
        return op(*inputs[0])
    if dim < 0:
        dim += len(shape_of(*inputs[0]))
    assert dim in (0, 1)
    notdim = 1 - dim
    new_size: Optional[Union[List[Expr], List[int]]] = None
    offsets_start = []
    offsets_end = []
    for i in range(len(inputs)):
        shape = shape_of(*inputs[i])
        if new_size is None:
            new_size = shape
        else:
            new_size[notdim] = V.graph.sizevars.guard_equals(shape[notdim], new_size[notdim])
            new_size[dim] += shape[dim]
        offsets_start.append(new_size[dim] - shape[dim])
        offsets_end.append(new_size[dim])
    assert new_size is not None
    dtype = functools.reduce(torch.promote_types, [x.get_dtype() for x in itertools.chain(*inputs)])
    device = inputs[0][0].get_device()
    kernel = ir.ConcatKernel(name=None, layout=ir.FixedLayout(device, dtype, new_size), inputs=[])
    kernel_tensor = ir.TensorBox.create(kernel)
    for i in range(len(inputs)):
        dst = ir.SliceView.create(kernel_tensor, dim, offsets_start[i], offsets_end[i])
        src = op(*inputs[i], layout=dst.get_layout()).data.data
        assert isinstance(src, (ir.ExternKernelOut, ir.TemplateBuffer))
        src.layout = ir.AliasedLayout(dst)
        kernel.inputs.append(src)
    kernel.name = V.graph.register_buffer(kernel)
    kernel.inputs = ir.ConcatKernel.unwrap_storage(kernel.inputs)
    return kernel_tensor
_cat_1 = CallFunction(aten.cat, Arg(), 1, _users=2)

@register_lowering_pattern(CallFunction(aten.cat, [_cat_1, CallFunction(aten.slice, _cat_1, 1, 0, KeywordArg('size'))], 1))
def cat_slice_cat(match, cat_input, size, dim=1):
    if False:
        i = 10
        return i + 15
    '\n    This is an example of a more complex pattern where cat_1 is used\n    multiple times inside the pattern.  We fold 2 calls to cat into one.\n\n    Matches:\n        cat_1: f32[1024, 4077] = torch.ops.aten.cat.default([add_26, primals_217], 1)\n        slice_1: f32[1024, 4077] = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)\n        slice_2: f32[1024, 19] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)\n        cat_2: f32[1024, 4096] = torch.ops.aten.cat.default([cat_1, slice_2], 1)\n\n\n    Rewrite to:\n        slice_2 = torch.ops.aten.slice.Tensor(add_26, 1, 0, 19)\n        cat_2 = torch.ops.aten.cat.default([add_26, primals_217, slice2], 1)\n    '
    (first, *rest) = cat_input
    if size >= 0 and V.graph.sizevars.statically_known_leq(size, first.get_size()[dim]):
        return L[aten.cat]([first, *rest, L[aten.slice](first, dim, 0, size)], dim)
    else:
        tmp = L[aten.cat](cat_input, dim)
        return L[aten.cat]([tmp, L[aten.slice](tmp, dim, 0, size)], dim)

def is_valid_splitwithsizes_cat(match):
    if False:
        for i in range(10):
            print('nop')
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    (split_node, cat_node) = (split_nodes[0], cat_nodes[0])
    if get_arg_value(split_node, 2, 'dim') != get_arg_value(cat_node, 1, 'dim'):
        return False
    get_item_args = {get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes}
    assert None not in get_item_args
    split_sizes = get_arg_value(split_node, 1, 'split_sizes')
    if get_item_args != set(range(len(split_sizes))):
        return False
    cat_items_args_order = [get_arg_value(item_node, 1) for item_node in get_arg_value(cat_node, 0)]
    if cat_items_args_order != list(range(len(split_sizes))):
        return False
    return True

def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    if False:
        print('Hello World!')
    'True if two nodes have the same metadata'
    val1 = node1.meta.get('val')
    val2 = node2.meta.get('val')
    return val1 is not None and val2 is not None and (val1.size() == val2.size()) and (val1.layout == val2.layout) and (val1.dtype == val2.dtype) and (val1.device == val2.device) and (val1.layout != torch.strided or val1.stride() == val2.stride())
noop_registry: Dict[Any, Any] = {}

def register_noop_decomp(targets, nop_arg=0):
    if False:
        return 10

    def register_fun(cond):
        if False:
            while True:
                i = 10
        register_decomposition(targets, registry=noop_registry, unsafe=True)((cond, nop_arg))
        return cond
    return register_fun

@register_noop_decomp(aten.slice)
def slice_noop(self, dim=0, start=None, end=None, step=1):
    if False:
        for i in range(10):
            print('nop')
    if start is None or end is None:
        return False
    if start == 0 and end >= 2 ** 63 - 1 and (step == 1):
        return True
    return False

@register_noop_decomp(aten.slice_scatter, 1)
def slice_scatter_noop(self, src, dim=0, start=None, end=None, step=1):
    if False:
        return 10
    if start is None:
        start = 0
    if end is None:
        end = 2 ** 63 - 1
    if start == 0 and end >= 2 ** 63 - 1 and (step == 1):
        return True
    return False

@register_noop_decomp(aten.repeat)
def repeat_noop(self, repeats):
    if False:
        while True:
            i = 10
    return all((r == 1 for r in repeats))

@register_noop_decomp(aten.constant_pad_nd)
def constant_pad_nd(x, padding, fill_value=0):
    if False:
        while True:
            i = 10
    return all((p == 0 for p in padding))

@register_noop_decomp(torch.ops.prims.convert_element_type)
def convert_element_type_noop(x, dtype: torch.dtype):
    if False:
        i = 10
        return i + 15
    return x.dtype == dtype

@register_noop_decomp(torch.ops.prims.device_put)
def device_put_noop(x, device):
    if False:
        i = 10
        return i + 15
    return x.device == decode_device(device)

@register_noop_decomp([aten.ceil, aten.floor, aten.round, aten.trunc])
def int_noop(x):
    if False:
        return 10
    return is_integer_dtype(x.dtype)

@register_noop_decomp([aten.pow])
def pow_noop(a, b):
    if False:
        i = 10
        return i + 15
    return isinstance(b, int) and b == 1

@register_noop_decomp([aten.cat], lambda args: args[0][0])
def cat_noop(inputs, dim=0):
    if False:
        while True:
            i = 10
    return len(inputs) == 1

@register_noop_decomp(aten.view)
def view_noop(arg, size):
    if False:
        i = 10
        return i + 15
    return arg.shape == size

@register_noop_decomp([aten.copy], nop_arg=1)
@register_noop_decomp([aten.alias, aten.clone])
def true_noop(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return True

def remove_noop_ops(graph: torch.fx.Graph):
    if False:
        return 10
    '\n    Removes both operations that are essentially aten.clone and operations that are essentially aten.alias from the graph.\n    '
    input_storages = set()
    output_storages = set()
    for node in graph.nodes:
        if node.op == 'placeholder':
            input_storages.add(get_node_storage(node))
        else:
            break
    for out in next(iter(reversed(graph.nodes))).args[0]:
        if isinstance(out, torch.fx.Node):
            output_storages.add(get_node_storage(out))
    for node in graph.nodes:
        if node.target in noop_registry:
            (cond, src_index) = noop_registry[node.target]
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                src = src_index(node.args)
            if not isinstance(src, torch.fx.Node):
                continue
            if get_node_storage(node) in output_storages and (get_node_storage(src) in input_storages or get_node_storage(src) in output_storages):
                continue
            (is_valid, args, kwargs) = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                graph.erase_node(node)
InplaceableOp = namedtuple('InplaceableOp', ['inplace_op', 'mutated_arg'])

def reinplace_inplaceable_ops(graph):
    if False:
        i = 10
        return i + 15
    "\n    Reinplaces in-placeable operations.\n    If there are no uses of a view of the mutated arg after the current node,\n    it is possible to inplace the op.\n    This above algorithm could be justified by observing side effects. While\n    we traverse the graph in forwards direction, only latter nodes could view\n    side effects of the current node. If the current node is not used later as\n    well as no view of this node is used later in the graph, then it is safe to\n    inplace as there would be no way to observe the side effects.\n    This condition is slightly different for graph inputs where they can only\n    be inplaced if the above condition is true and there's a copy_ in the\n    epilogue that signals that the caller wants to observe the mutation.\n    "
    copy_args_to_copy_nodes = {}
    mutated_inputs = set()
    storage_to_nodes = defaultdict(list)
    node_order: Dict[Any, int] = {}
    for (i, node) in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage_to_nodes[get_node_storage(node)].append(node)
        if node.target == aten.copy_.default:
            dst = node.args[0]
            src = node.args[1]
            if src.target == operator.getitem and src.args[0].target == triton_kernel_wrapper_functional and (src.args[0].kwargs['kwargs'][src.args[1]] == node.args[0]):
                src = src.args[0]
            copy_args_to_copy_nodes[dst, src] = node
            assert node.args[0].op == 'placeholder'
            mutated_inputs.add(node.args[0])

    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node):
        if False:
            for i in range(10):
                print('nop')
        node_loc = node_order[node]
        for view in shared_view_nodes:
            for user in view.users:
                if node_order[user] <= node_loc:
                    continue
                if copy_node == user:
                    continue
                return True
        return False

    def can_inplace(node, mutated_arg):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(mutated_arg, (list, tuple)):
            return all((can_inplace(node, arg) for arg in mutated_arg))
        if get_node_storage(mutated_arg) is None:
            return False
        shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]
        if mutated_arg.op == 'placeholder':
            if not (copy_node := copy_args_to_copy_nodes.get((mutated_arg, node), False)):
                return False
            if any_use_of_views_after_node(node, shared_view_nodes, copy_node=copy_node):
                return False
            return True
        elif any((view.op == 'placeholder' for view in shared_view_nodes)):
            return False
        else:
            return not any_use_of_views_after_node(node, shared_view_nodes, copy_node=None)
    inplaceable_ops = {aten.index_put.default: InplaceableOp(aten.index_put_.default, 0), aten._unsafe_index_put.default: InplaceableOp(inductor_prims._unsafe_index_put_, 0)}
    try:
        c10d_functional = torch.ops._c10d_functional
        inplaceable_collective_ops = {c10d_functional.all_reduce.default: InplaceableOp(c10d_functional.all_reduce_.default, 0), c10d_functional.all_reduce_coalesced.default: InplaceableOp(c10d_functional.all_reduce_coalesced_.default, 0)}
        inplaceable_ops.update(inplaceable_collective_ops)
    except AttributeError:
        pass
    inplaceable_triton_ops = {triton_kernel_wrapper_functional}
    for node in graph.nodes:
        if (inplaceable_op := inplaceable_ops.get(node.target, None)) is not None:
            mutated_arg = node.args[inplaceable_op.mutated_arg]
            if can_inplace(node, mutated_arg):
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    graph.erase_node(copy_node)
                node.target = inplaceable_op.inplace_op
        elif node.target in inplaceable_triton_ops:
            tensors_to_clone = []
            for arg in node.kwargs['tensors_to_clone']:
                assert arg in node.kwargs['kwargs']
                mutated_arg = node.kwargs['kwargs'][arg]
                if can_inplace(node, mutated_arg):
                    copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                    if copy_node is not None:
                        graph.erase_node(copy_node)
                else:
                    tensors_to_clone.append(arg)
            kwargs = dict(node.kwargs)
            kwargs['tensors_to_clone'] = tensors_to_clone
            node.kwargs = immutable_dict(kwargs)

@register_lowering_pattern(CallFunction(aten.cat, ListOf(CallFunction(operator.getitem, CallFunction(aten.split_with_sizes, KeywordArg('input_'), Ignored(), Ignored(), _users=MULTIPLE), Ignored())), Ignored()), pass_number=2, extra_check=is_valid_splitwithsizes_cat)
def splitwithsizes_cat_replace(match, input_):
    if False:
        print('Hello World!')
    return input_

def is_valid_cat_splitwithsizes(match):
    if False:
        return 10
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    (split_node, cat_node) = (split_nodes[0], cat_nodes[0])
    if len(cat_node.users) > 1:
        return False
    dim = get_arg_value(split_node, 2, 'dim')
    if dim != get_arg_value(cat_node, 1, 'dim'):
        return False
    cat_inputs = list(get_arg_value(cat_node, 0))
    split_sizes = get_arg_value(split_node, 1, 'split_sizes')
    if len(cat_inputs) != len(split_sizes):
        return False
    for (cat_input, split_size) in zip(cat_inputs, split_sizes):
        if 'val' not in cat_input.meta:
            return False
        cat_input_size = cat_input.meta['val'].size(dim)
        if cat_input_size != split_size:
            return False
    return True

@register_lowering_pattern(CallFunction(aten.split_with_sizes, CallFunction(aten.cat, KeywordArg('input_'), Ignored(), _users=MULTIPLE), Ignored(), Ignored()), pass_number=2, extra_check=is_valid_cat_splitwithsizes)
def cat_splitwithsizes_replace(match, input_):
    if False:
        for i in range(10):
            print('nop')
    return input_

def view_to_reshape(gm):
    if False:
        for i in range(10):
            print('nop')
    '\n    Replace view ops in the GraphModule to reshape ops.\n    '
    for nd in gm.graph.nodes:
        if nd.target == torch.ops.aten.view.default:
            nd.target = torch.ops.aten.reshape.default

def should_prefer_unfused_addmm(match):
    if False:
        i = 10
        return i + 15
    inp = match.kwargs['inp']
    if not inp.meta['val'].is_cuda:
        return False
    output = match.output_node()
    return all((is_pointwise_use(use) for use in output.users))

@register_graph_pattern(CallFunction(aten.addmm, KeywordArg('inp'), Arg(), Arg()), pass_dict=pass_patterns[2], extra_check=should_prefer_unfused_addmm)
def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp):
    if False:
        for i in range(10):
            print('nop')

    def repl(inp, x1, x2):
        if False:
            print('Hello World!')
        return x1 @ x2 + inp
    with V.fake_mode:
        match.replace_by_example(repl, [inp, mat1, mat2])

def is_valid_addmm_fusion(match):
    if False:
        while True:
            i = 10
    (mat1, mat2) = match.args
    inp = match.kwargs['inp']
    if not (isinstance(inp, torch.fx.Node) and isinstance(inp.meta['val'], torch.Tensor)):
        return False
    in_shape = inp.meta['val'].shape
    mm_shape = (mat1.meta['val'].shape[0], mat2.meta['val'].shape[1])
    matched = is_expandable_to(in_shape, mm_shape)
    if not matched:
        return False
    return not should_prefer_unfused_addmm(match)

@register_graph_pattern(CallFunction(aten.add, CallFunction(aten.mm, Arg(), Arg()), KeywordArg('inp')), pass_dict=pass_patterns[2], extra_check=is_valid_addmm_fusion)
@register_graph_pattern(CallFunction(aten.add, KeywordArg('inp'), CallFunction(aten.mm, Arg(), Arg())), pass_dict=pass_patterns[2], extra_check=is_valid_addmm_fusion)
def addmm(match, mat1, mat2, *, inp):
    if False:
        i = 10
        return i + 15

    def repl(inp, mat1, mat2):
        if False:
            i = 10
            return i + 15
        return aten.addmm(inp, mat1, mat2)
    with V.fake_mode:
        match.replace_by_example(repl, [inp, mat1, mat2])

def check_shape_cuda_and_fused_int_mm_mul_enabled(match):
    if False:
        for i in range(10):
            print('nop')
    return config.force_fuse_int_mm_with_mul and len(getattr(match.args[2].meta.get('val'), 'shape', [])) == 2 and getattr(match.args[2].meta.get('val'), 'is_cuda', False)

@register_lowering_pattern(CallFunction(prims.convert_element_type.default, CallFunction(aten.mul, CallFunction(aten._int_mm, Arg(), Arg()), Arg()), Arg()), check_shape_cuda_and_fused_int_mm_mul_enabled)
@register_lowering_pattern(CallFunction(aten.mul, CallFunction(aten._int_mm, Arg(), Arg()), Arg()), check_shape_cuda_and_fused_int_mm_mul_enabled)
def fused_int_mm_mul(match: Match, mat1, mat2, mat3, out_dtype=None):
    if False:
        for i in range(10):
            print('nop')
    return inductor.kernel.mm.tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype)