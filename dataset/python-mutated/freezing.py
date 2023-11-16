from __future__ import annotations
import itertools
import logging
import weakref
from typing import Any, List, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.constant_folding import constant_fold, replace_node_with_constant
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape
from . import config
aten = torch.ops.aten
prims = torch.ops.prims
log = logging.getLogger(__name__)

def replace_params_with_constants(gm: torch.fx.GraphModule, flat_params: list[Any], fw_metadata: torch._functorch.aot_autograd.ViewAndMutationMeta) -> List[int]:
    if False:
        i = 10
        return i + 15
    '\n    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.\n    Returns a list of indices representing the input parameters that were not converted to constants.\n    '
    params = [node for node in gm.graph.nodes if node.op == 'placeholder']
    fake_inp_nodes = params[:len(params)]
    preserved_arg_indices = []
    aliased_input_args = [out_info.base_idx for out_info in fw_metadata.output_info if out_info.base_idx is not None]
    for (i, (real_input, node)) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in fw_metadata.mutated_inp_indices or i in aliased_input_args:
            preserved_arg_indices.append(i)
            continue
        replace_node_with_constant(gm, node, real_input)
    preserved_arg_indices.extend(range(len(flat_params), len(params)))
    gm.recompile()
    return preserved_arg_indices

def freeze(dynamo_gm: torch.fx.GraphModule, aot_autograd_gm: torch.fx.GraphModule, example_inputs: List[torch._subclasses.FakeTensor]) -> Tuple[torch.fx.GraphModule, List[int]]:
    if False:
        while True:
            i = 10
    '\n    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation\n    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.\n\n    Assumes that this function is run in dynamo tracing post aot_autograd.\n\n    Args:\n        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.\n        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.\n        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.\n\n    Returns:\n        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices\n        of the inputs that were preserved (not turned into constants).\n    '
    view_to_reshape(aot_autograd_gm)
    if (tracing_context := torch._guards.TracingContext.try_get()):
        fw_metadata = tracing_context.fw_metadata
        params_flat = tracing_context.params_flat
        assert fw_metadata is not None and params_flat is not None
        preserved_arg_indices = replace_params_with_constants(aot_autograd_gm, params_flat, fw_metadata)
    else:
        inputs = [node for node in aot_autograd_gm.graph.nodes if node.op == 'placeholder']
        preserved_arg_indices = list(range(len(inputs)))
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm.recompile()
    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
    freezing_passes(aot_autograd_gm, aot_example_inputs)
    constant_fold(aot_autograd_gm)
    if config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)
    log.debug('%s', lazy_format_graph_code('FROZEN GRAPH', aot_autograd_gm))
    return (aot_autograd_gm, preserved_arg_indices)

class ErasedTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        if False:
            while True:
                i = 10
        return super().__new__(cls, elem.to(device='meta'))

    def __init__(self, elem, name: Optional[str], mod):
        if False:
            i = 10
            return i + 15
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        erased_tensors = [e for e in pytree.arg_tree_leaves(*args, **kwargs) if isinstance(e, ErasedTensor)]
        assert len(erased_tensors) > 0
        e = erased_tensors[0]
        raise RuntimeError(f'Trying to run Pytorch Eager Module after Dynamo Freezing. The original parameters have been discarded for memory efficiency. Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}')

@torch.utils._python_dispatch._disable_current_modes()
def invalidate_eager_modules():
    if False:
        return 10
    for mod in torch._guards.TracingContext.get().module_context.nn_modules.values():
        if not isinstance(mod, torch.nn.Module):
            continue
        for (attr_name, tensor) in list(itertools.chain(mod.named_parameters(recurse=False), mod.named_buffers(recurse=False))):
            with torch._dispatch.python.no_python_dispatcher():
                e_t = ErasedTensor(tensor, attr_name, mod)
            if isinstance(tensor, torch.nn.Parameter):
                e_t.requires_grad_(True)
                e_t._is_param = True
            setattr(mod, attr_name, e_t)

@torch.utils._python_dispatch._disable_current_modes()
def discard_traced_gm_params(mod: torch.fx.GraphModule):
    if False:
        print('Hello World!')
    for (attr_name, tensor) in list(itertools.chain(mod.named_parameters(recurse=False), mod.named_buffers(recurse=False))):
        with torch._dispatch.python.no_python_dispatcher():
            e_t = ErasedTensor(tensor, attr_name, mod)
        if isinstance(tensor, torch.nn.Parameter):
            e_t.requires_grad_(True)
            e_t._is_param = True
        setattr(mod, attr_name, e_t)

def enforce_output_layout(gm: torch.fx.GraphModule):
    if False:
        for i in range(10):
            print('nop')
    "\n    Make sure the output node's layout does not change due to compiler optimizations\n    by adding aten.as_strided nodes with the expected strides.\n\n    Only used for inference so we can assume all graph outputs are model outputs.\n    "
    (*_, output_node) = gm.graph.nodes
    out_list = output_node.args[0]
    with gm.graph.inserting_before(output_node):
        for n in out_list:
            if not isinstance(n.meta['val'], torch.Tensor) or not torch._prims_common.is_non_overlapping_and_dense(n.meta['val']):
                continue
            ft = n.meta['val']
            new_node = gm.graph.call_function(prims.inductor_force_stride_order.default, (n, ft.stride()))
            output_node.replace_input_with(n, new_node)
    gm.graph.lint()
    gm.recompile()

def enforce_as_strided_input_layout(gm: torch.fx.GraphModule):
    if False:
        print('Hello World!')
    "\n    Make sure the as_strided node's input's layout does not change due to compiler\n    optimizations, because the as_strided strides info depends on input tensor stride info.\n    "
    as_strided_ops = [torch.ops.aten.as_strided.default, torch.ops.aten.as_strided_.default, torch.ops.aten.as_strided_scatter.default]
    strided_nodes = [n for n in gm.graph.nodes if n.target in as_strided_ops]
    for n in strided_nodes:
        with gm.graph.inserting_before(n):
            ft = n.args[0].meta['val']
            new_node = gm.graph.call_function(prims.inductor_force_stride_order.default, (n.args[0], ft.stride()))
            n.replace_input_with(n.args[0], new_node)
    gm.graph.lint()
    gm.recompile()

@dynamo_timed
def convert_conv_weights_to_channels_last(gm: torch.fx.GraphModule):
    if False:
        print('Hello World!')
    '\n    Convert 4d convolution weight tensor to channels last format.\n\n    This pass is performed before freezing so the added nodes can be constant\n    folded by freezing.\n    '
    convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
    for conv in convs:
        weight_node = conv.args[1]
        if len(weight_node.meta['val'].size()) != 4 or weight_node.meta['val'].is_contiguous(memory_format=torch.channels_last):
            continue
        with gm.graph.inserting_before(conv):
            new_node = gm.graph.call_function(aten.clone.default, (weight_node,), {'memory_format': torch.channels_last})
            conv.replace_input_with(weight_node, new_node)
    enforce_as_strided_input_layout(gm)
    enforce_output_layout(gm)