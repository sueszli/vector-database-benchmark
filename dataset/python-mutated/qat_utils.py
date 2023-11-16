import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import replace_pattern_with_filters, ReplacedPatterns
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.ao.quantization.quantizer import DerivedQuantizationSpec, EdgeOrNode, SharedQuantizationSpec, QuantizationSpecBase
from .utils import _is_supported_batch_norm_for_training, fold_bn_weights_into_conv_node, get_aten_graph_module
__all__ = []
_conv2d_bn_pattern_example_inputs = (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 1, 1), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1))
_quantized_conv2d_bn_pattern_example_inputs = (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 1, 1), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1))

def _get_quantized_conv2d_bn_pattern_example_inputs_kwargs(is_per_channel: bool, has_bias: bool, is_cuda: bool) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    '\n    Optional example inputs for both `_quantized_qat_conv2d_bn_pattern`\n    and `_folded_quantized_qat_conv2d_bn_pattern`, expressed as kwargs.\n\n    '
    kwargs = {}
    if is_per_channel:
        kwargs['scale'] = torch.tensor([1], dtype=torch.float)
        kwargs['zero_point'] = torch.tensor([0], dtype=torch.int)
    if has_bias:
        kwargs['conv_bias'] = torch.randn(1)
    if is_cuda:
        for (k, v) in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.cuda()
    return kwargs

def _conv2d_bn_pattern(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor) -> torch.Tensor:
    if False:
        print('Hello World!')
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True)
    return x

def _qat_conv2d_bn_pattern(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    '\n    Approximated method to fuse conv and bn. It requires only one forward pass.\n    conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std.\n    This is based on `nniqat.ConvBn2d._forward_approximate`.\n    '
    bn_eps = 1e-05
    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
    x = F.conv2d(x, scaled_weight, zero_bias)
    x = x / scale_factor.reshape(bias_shape)
    x = x + conv_bias.reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    return x

def _qat_conv2d_bn_pattern_no_conv_bias(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor) -> torch.Tensor:
    if False:
        print('Hello World!')
    '\n    Same as `_qat_conv2d_bn_pattern`, but handles the case with no conv bias.\n    '
    bn_eps = 1e-05
    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    x = F.conv2d(x, scaled_weight, None)
    x = x / scale_factor.reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    return x

def _append_qdq(x, is_per_channel, kwargs):
    if False:
        while True:
            i = 10
    '\n    Helper function to append q-dq ops after `x`, using dummy values for the qparams\n    and qmin/qmax. We use dummy values here because we match with `ignore_literals=True`\n    and will manually replace these values after subgraph rewriting.\n\n    Return the dq node.\n    '
    per_channel_axis = 0
    scale = kwargs['scale'] if is_per_channel else 1.0
    zp = kwargs['zero_point'] if is_per_channel else 0
    qmin = -127
    qmax = 127
    dtype = torch.int8
    qd = torch.ops.quantized_decomposed
    if is_per_channel:
        x = qd.quantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
        x = qd.dequantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
    else:
        x = qd.quantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
        x = qd.dequantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
    return x

def _get_quantized_qat_conv2d_bn_pattern(is_per_channel: bool, has_bias: bool, bias_is_quantized: bool) -> Callable:
    if False:
        print('Hello World!')
    '\n    Return the quantized version of QAT conv + BN pattern.\n    This is based on `nniqat.ConvBn2d._forward_approximate`,\n    used in QAT convert. We first match this pattern and replace\n    it with the normal [conv - bn] pattern, then fold the BN\n    weights into conv.\n    '
    bn_eps = 1e-05

    def _quantized_qat_conv2d_bn_pattern(x: torch.Tensor, conv_weight: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor, **kwargs) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        running_std = torch.sqrt(bn_running_var + bn_eps)
        scale_factor = bn_weight / running_std
        weight_shape = [1] * len(conv_weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(conv_weight.shape)
        bias_shape[1] = -1
        scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
        scaled_weight = _append_qdq(scaled_weight, is_per_channel, kwargs)
        if has_bias:
            zero_bias = torch.zeros_like(kwargs['conv_bias'], dtype=x.dtype)
            if bias_is_quantized:
                zero_bias = _append_qdq(zero_bias, is_per_channel, kwargs)
            x = F.conv2d(x, scaled_weight, zero_bias)
        else:
            x = F.conv2d(x, scaled_weight, None)
        x = x / scale_factor.reshape(bias_shape)
        if has_bias:
            x = x + kwargs['conv_bias'].reshape(bias_shape)
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
        return x
    return _quantized_qat_conv2d_bn_pattern

def _get_folded_quantized_qat_conv2d_bn_pattern(is_per_channel: bool, has_bias: bool, bias_is_quantized: bool) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Quantized QAT conv - bn pattern with bn weights being folded into conv.\n    '
    bn_eps = 1e-05

    def _folded_quantized_qat_conv2d_bn_pattern(x: torch.Tensor, conv_weight: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor, **kwargs) -> torch.Tensor:
        if False:
            return 10
        conv_weight = _append_qdq(conv_weight, is_per_channel, kwargs)
        if has_bias:
            bias = kwargs['conv_bias']
            if bias_is_quantized:
                bias = _append_qdq(bias, is_per_channel, kwargs)
        else:
            bias = None
        x = F.conv2d(x, conv_weight, bias)
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
        return x
    return _folded_quantized_qat_conv2d_bn_pattern

def _has_conv_bias_filter(match: 'InternalMatch', original_graph: Graph, pattern_graph: Graph) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Match filter for the subgraph rewriter that returns True if the conv node in\n    the original graph has bias.\n    '
    for n in match.nodes_map.values():
        if n.target == torch.ops.aten.conv2d.default:
            return len(n.args) > 2 and n.args[2] is not None
    raise ValueError('Could not find conv node in matched conv + bn pattern')

def _no_conv_bias_filter(match: 'InternalMatch', original_graph: Graph, pattern_graph: Graph) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Match filter for the subgraph rewriter that returns True if the conv node in\n    the original graph does NOT have bias.\n    '
    return not _has_conv_bias_filter(match, original_graph, pattern_graph)

def _is_quantize(n: Node) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return n.target in [torch.ops.quantized_decomposed.quantize_per_tensor.default, torch.ops.quantized_decomposed.quantize_per_tensor.tensor, torch.ops.quantized_decomposed.quantize_per_channel.default]

def _is_dequantize(n: Node) -> bool:
    if False:
        print('Hello World!')
    return n.target in [torch.ops.quantized_decomposed.dequantize_per_tensor.default, torch.ops.quantized_decomposed.dequantize_per_tensor.tensor, torch.ops.quantized_decomposed.dequantize_per_channel.default]

def _get_conv_bn_pattern_nodes(r: ReplacedPatterns) -> Dict[str, Tuple[Node, Node]]:
    if False:
        i = 10
        return i + 15
    '\n    Helper function to extract the nodes in the conv-bn fusion pattern after\n    subgraph rewriting, in the form of a map:\n\n        {name: (original_node, replacement_node)}\n\n    The following names must exist in the map:\n\n        "conv", "conv_weight", "conv_input", "bn", "getitem"\n\n    The following names may exist in the map:\n\n        "conv_weight_q", "conv_weight_dq", "conv_bias",\n        "conv_bias_q", "conv_bias_dq"\n    '

    def _get_nodes(nodes: List[Node]) -> Tuple[Node, Node, Node]:
        if False:
            print('Hello World!')
        '\n        Return a 3-tuple of (conv_node, bn_node, getitem_node).\n        This asserts that the match contains exactly one of each node.\n        '
        (conv_node, bn_node, getitem_node) = (None, None, None)
        for n in nodes:
            if n.op != 'call_function':
                continue
            if n.target == torch.ops.aten.conv2d.default:
                assert conv_node is None
                conv_node = n
            if _is_supported_batch_norm_for_training(n):
                assert bn_node is None
                bn_node = n
            if n.target == operator.getitem:
                assert getitem_node is None
                getitem_node = n
        assert conv_node is not None
        assert bn_node is not None
        assert getitem_node is not None
        return (conv_node, bn_node, getitem_node)

    def _get_q_dq_nodes(n: Node) -> Tuple[Node, Node, Node]:
        if False:
            return 10
        '\n        Return a 3-tuple of (orig_node, q_node, dq_node).\n        '
        assert _is_dequantize(n)
        q_node = n.args[0]
        assert isinstance(q_node, Node)
        assert _is_quantize(q_node)
        orig_node = q_node.args[0]
        assert isinstance(orig_node, Node)
        return (orig_node, q_node, n)
    original_nodes = list(_filter_nodes_map(r.nodes_map).values())
    (o_conv, o_bn, o_getitem) = _get_nodes(original_nodes)
    (r_conv, r_bn, r_getitem) = _get_nodes(r.replacements)
    mapping = {'conv': (o_conv, r_conv), 'bn': (o_bn, r_bn), 'getitem': (o_getitem, r_getitem)}
    (p_conv, _, _) = _get_nodes(list(r.nodes_map.keys()))
    (p_conv_input, p_conv_weight, *_) = p_conv.args
    (r_conv_input, r_conv_weight, *_) = r_conv.args
    assert isinstance(p_conv_input, Node)
    assert isinstance(p_conv_weight, Node)
    assert isinstance(r_conv_input, Node)
    assert isinstance(r_conv_weight, Node)
    o_conv_input = r.nodes_map[p_conv_input]
    o_conv_weight = r.nodes_map[p_conv_weight]
    if _is_dequantize(p_conv_weight):
        (p_conv_weight, p_conv_weight_q, p_conv_weight_dq) = _get_q_dq_nodes(p_conv_weight)
        (r_conv_weight, r_conv_weight_q, r_conv_weight_dq) = _get_q_dq_nodes(r_conv_weight)
        o_conv_weight = r.nodes_map[p_conv_weight]
        o_conv_weight_q = r.nodes_map[p_conv_weight_q]
        o_conv_weight_dq = r.nodes_map[p_conv_weight_dq]
        mapping['conv_weight_q'] = (o_conv_weight_q, r_conv_weight_q)
        mapping['conv_weight_dq'] = (o_conv_weight_dq, r_conv_weight_dq)
    mapping['conv_input'] = (o_conv_input, r_conv_input)
    mapping['conv_weight'] = (o_conv_weight, r_conv_weight)
    if len(p_conv.args) > 2 and len(r_conv.args) > 2:
        p_conv_bias = p_conv.args[2]
        r_conv_bias = r_conv.args[2]
        assert isinstance(p_conv_bias, Node)
        assert isinstance(r_conv_bias, Node)
        o_conv_bias = r.nodes_map[p_conv_bias]
        if _is_dequantize(p_conv_bias):
            (p_conv_bias, p_conv_bias_q, p_conv_bias_dq) = _get_q_dq_nodes(p_conv_bias)
            (r_conv_bias, r_conv_bias_q, r_conv_bias_dq) = _get_q_dq_nodes(r_conv_bias)
            o_conv_bias = r.nodes_map[p_conv_bias]
            o_conv_bias_q = r.nodes_map[p_conv_bias_q]
            o_conv_bias_dq = r.nodes_map[p_conv_bias_dq]
            mapping['conv_bias_q'] = (o_conv_bias_q, r_conv_bias_q)
            mapping['conv_bias_dq'] = (o_conv_bias_dq, r_conv_bias_dq)
        mapping['conv_bias'] = (o_conv_bias, r_conv_bias)
    return mapping

def _filter_nodes_map(nodes_map: Dict[Node, Node]) -> Dict[Node, Node]:
    if False:
        return 10
    '\n    Return a filtered `nodes_map` returned from the subgraph rewriter.\n    The filtered `nodes_map` will contain only nodes that are actually\n    matched in the pattern, excluding None or placeholder nodes.\n    '
    new_nodes_map: Dict[Node, Node] = {}
    for (pattern_node, graph_node) in nodes_map.items():
        if graph_node is None:
            continue
        if pattern_node.op == 'placeholder':
            continue
        new_nodes_map[pattern_node] = graph_node
    return new_nodes_map

def _copy_over_literal_conv_args(original_node: Node, new_node: Node):
    if False:
        print('Hello World!')
    '\n    Copy over literal args in conv, such as stride and padding, from the matched node\n    in the original graph to its replacement in the new graph.\n\n    This is needed due to the following limitation in the subgraph rewriter when used\n    with dynamo export: literal (non-tensor) args are not supported in the match and\n    replacement patterns. This is because dynamo export automatically inlines these\n    literal args, making them dead placeholder nodes. In the future, we should check\n    if dynamo export can optionally disable this inlining, or if subgraph rewriter\n    can do the copying for us. See https://github.com/pytorch/pytorch/issues/100419.\n\n    Note: Unlike other tensor args like conv weights and biases, literal args are\n    preserved in the original nodes after replacement, so we can access them here.\n    '
    assert original_node.target == torch.ops.aten.conv2d.default
    assert new_node.target == torch.ops.aten.conv2d.default
    new_args = list(new_node.args)
    if len(new_args) < 3:
        new_args.append(None)
    new_node.args = tuple(new_args[:3]) + original_node.args[3:]

def _update_conv_input_qspec_map_after_replacement(original_node: Node, replacement_node: Node):
    if False:
        print('Hello World!')
    '\n    Update the `input_qspec_map` in the annotation after subgraph rewriting.\n\n    The original annotation referred to the nodes in the original graph,\n    so the keys in the `input_qspec_map` will need to be updated to reflect\n    the corresponding nodes in the replacement graph.\n    '
    assert original_node.target == torch.ops.aten.conv2d.default
    assert replacement_node.target == torch.ops.aten.conv2d.default
    if 'quantization_annotation' not in original_node.meta:
        return
    original_input_qspec_map = original_node.meta['quantization_annotation'].input_qspec_map
    input_qspec_map = {}
    all_configs = list(original_input_qspec_map.items())
    input_qspec_map[replacement_node.args[0]] = all_configs[0][1]
    input_qspec_map[replacement_node.args[1]] = all_configs[1][1]
    if len(replacement_node.args) > 2 and len(all_configs) > 2:
        input_qspec_map[replacement_node.args[2]] = all_configs[2][1]
    replacement_node.meta['quantization_annotation'].input_qspec_map = input_qspec_map

def _update_special_qspecs_after_replacement(node: Node, original_to_replacement_node: Dict[Node, Node]):
    if False:
        return 10
    "\n    Update the `SharedQuantizationSpec`s and `DerivedQuantizationSpec`s\n    used in `node`'s quantization annotation after subgraph rewriting.\n\n    The original annotation referred to the nodes in the original graph,\n    so the nodes used in these special quantization specs will need to\n    be updated to the corresponding nodes in the replacement graph.\n    "

    def _get_new_edge_or_node(edge_or_node: EdgeOrNode):
        if False:
            while True:
                i = 10
        if isinstance(edge_or_node, Node):
            _node = edge_or_node
            return original_to_replacement_node.get(_node, _node)
        elif isinstance(edge_or_node, tuple) and len(edge_or_node) == 2 and all((isinstance(x, Node) for x in edge_or_node)):
            (src, dest) = edge_or_node
            return (original_to_replacement_node.get(src, src), original_to_replacement_node.get(dest, dest))
        else:
            raise ValueError('unexpected type for edge_or_node: ', type(edge_or_node))

    def _get_new_qspec(qspec: QuantizationSpecBase):
        if False:
            return 10
        if isinstance(qspec, SharedQuantizationSpec):
            new_edge_or_node = _get_new_edge_or_node(qspec.edge_or_node)
            return SharedQuantizationSpec(new_edge_or_node)
        elif isinstance(qspec, DerivedQuantizationSpec):
            new_derived_from = [_get_new_edge_or_node(x) for x in qspec.derived_from]
            return dataclasses.replace(qspec, derived_from=new_derived_from)
        else:
            return qspec
    if 'quantization_annotation' not in node.meta:
        return
    annotation = node.meta['quantization_annotation']
    for (input_node, qspec) in annotation.input_qspec_map.items():
        annotation.input_qspec_map[input_node] = _get_new_qspec(qspec)
    annotation.output_qspec = _get_new_qspec(annotation.output_qspec)

def _fuse_conv_bn_qat(m: GraphModule) -> GraphModule:
    if False:
        i = 10
        return i + 15
    m = _fuse_conv_bn_qat_helper(m, is_cuda=False)
    if torch.cuda.is_available():
        m = _fuse_conv_bn_qat_helper(m, is_cuda=True)
    return m

def _fuse_conv_bn_qat_helper(m: GraphModule, is_cuda: bool) -> GraphModule:
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with\n    the fused QAT subgraph equivalent. The input graph should already be annotated.\n    The annotations in the original nodes will be preserved in the corresponding\n    nodes in the new subgraph.\n\n    Note: This also handles the (conv + bn + relu) pattern.\n    '
    m.graph.eliminate_dead_code()
    m.recompile()
    example_inputs = _conv2d_bn_pattern_example_inputs
    match_pattern = get_aten_graph_module(_conv2d_bn_pattern, example_inputs, is_cuda)
    replacement_pattern_with_conv_bias = get_aten_graph_module(_qat_conv2d_bn_pattern, example_inputs, is_cuda)
    replacements_with_conv_bias = replace_pattern_with_filters(m, match_pattern, replacement_pattern_with_conv_bias, match_filters=[_has_conv_bias_filter], ignore_literals=True)
    m.recompile()
    replacement_pattern_no_conv_bias = get_aten_graph_module(_qat_conv2d_bn_pattern_no_conv_bias, example_inputs, is_cuda)
    replacements_no_conv_bias = replace_pattern_with_filters(m, match_pattern, replacement_pattern_no_conv_bias, match_filters=[_no_conv_bias_filter], ignore_literals=True)
    m.recompile()
    all_original_to_replacement_nodes = {}
    for r in replacements_with_conv_bias + replacements_no_conv_bias:
        for (original_node, replacement_node) in _get_conv_bn_pattern_nodes(r).values():
            replacement_node.meta = original_node.meta
            if original_node.target == torch.ops.aten.conv2d.default:
                _copy_over_literal_conv_args(original_node, replacement_node)
                _update_conv_input_qspec_map_after_replacement(original_node, replacement_node)
            all_original_to_replacement_nodes[original_node] = replacement_node
    for n in m.graph.nodes:
        _update_special_qspecs_after_replacement(n, all_original_to_replacement_nodes)
    return m

def _duplicate_dequantize_node(m: GraphModule):
    if False:
        while True:
            i = 10
    '\n    Helper function to duplicate all dequantize nodes in the graph if the\n    node has more than one user. For example:\n\n    Before:\n      quantize -> dequantize -> a\n                          \\--> b\n                          \\--> c\n\n    After:\n      quantize -> dequantize_1 -> a\n            \\--> dequantize_2 -> b\n            \\--> dequantize_3 -> c\n\n    This is useful for subgraph rewriting. E.g. if we wish to match the\n    pattern [dequantize - a] above, subgraph matching would fail because\n    the dequantize node has users outside the matched portion of the graph.\n    Instead, we match [dequantize_1 - a], which is safe.\n    '
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    for n in m.graph.nodes:
        if n.op != 'call_function' or n.target != dq_op or len(n.users) == 1:
            continue
        for user in list(n.users):
            with m.graph.inserting_before(n):
                new_node = m.graph.create_node('call_function', dq_op, n.args, n.kwargs)
            user.replace_input_with(n, new_node)
        m.graph.erase_node(n)
    m.recompile()

def _remove_extra_dequantize(m: GraphModule):
    if False:
        for i in range(10):
            print('nop')
    '\n    Removes duplicate dequant nodes in the graph, for an operator that has\n    multiple dequant nodes as a user, replace them with a single dequant node\n    that can be shared across all the uses. This should be seen as the "reverse"\n    of `_duplicate_dequantize_node`.\n    '
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    for n in m.graph.nodes:
        dq_users = [user for user in n.users if user.op == 'call_function' and user.target == dq_op]
        if len(dq_users) > 1:
            with m.graph.inserting_after(dq_users[0]):
                new_node = m.graph.create_node('call_function', dq_op, dq_users[0].args, {})
            for dq_user in dq_users:
                dq_user.replace_all_uses_with(new_node)
                m.graph.erase_node(dq_user)
    m.recompile()

def _copy_over_q_dq_args(original_node: Node, replacement_node: Node):
    if False:
        i = 10
        return i + 15
    '\n    Given a pair of quantize or dequantize nodes, copy over all literal args\n    from the original node to the replacement node.\n    '
    assert original_node.target == replacement_node.target
    if original_node.target in (torch.ops.quantized_decomposed.quantize_per_tensor.default, torch.ops.quantized_decomposed.dequantize_per_tensor.default):
        start_copy_arg_index = 1
    elif original_node.target in (torch.ops.quantized_decomposed.quantize_per_channel.default, torch.ops.quantized_decomposed.dequantize_per_channel.default):
        start_copy_arg_index = 3
    else:
        raise ValueError("Expected quantize/dequantize nodes, got '%s'" % original_node.target)
    replacement_node.args = replacement_node.args[:start_copy_arg_index] + original_node.args[start_copy_arg_index:]

def _fold_conv_bn_qat(m: GraphModule) -> GraphModule:
    if False:
        print('Hello World!')
    m = _fold_conv_bn_qat_helper(m, is_cuda=False)
    if torch.cuda.is_available():
        m = _fold_conv_bn_qat_helper(m, is_cuda=True)
    return m

def _fold_conv_bn_qat_helper(m: GraphModule, is_cuda: bool) -> GraphModule:
    if False:
        i = 10
        return i + 15
    '\n    Replace the quantized (conv + bn) pattern with conv with bn weights folded into the weights of conv.\n    '
    m.graph.eliminate_dead_code()
    m.recompile()
    _duplicate_dequantize_node(m)
    replacements = []
    replacement_options = itertools.product([True, False], [True, False], [True, False])
    for (is_per_channel, has_bias, bias_is_quantized) in replacement_options:
        if not has_bias and bias_is_quantized:
            continue
        example_inputs = _quantized_conv2d_bn_pattern_example_inputs
        kwargs = _get_quantized_conv2d_bn_pattern_example_inputs_kwargs(is_per_channel, has_bias, is_cuda)
        match_pattern = _get_quantized_qat_conv2d_bn_pattern(is_per_channel, has_bias, bias_is_quantized)
        match_pattern = get_aten_graph_module(match_pattern, example_inputs, is_cuda, **kwargs)
        replacement_pattern = _get_folded_quantized_qat_conv2d_bn_pattern(is_per_channel, has_bias, bias_is_quantized)
        replacement_pattern = get_aten_graph_module(replacement_pattern, example_inputs, is_cuda, **kwargs)
        replacements.extend(replace_pattern_with_filters(m, match_pattern, replacement_pattern, ignore_literals=True))
    m.recompile()
    _remove_extra_dequantize(m)
    for r in replacements:
        node_map = _get_conv_bn_pattern_nodes(r)
        for (original_node, replacement_node) in node_map.values():
            replacement_node.meta = original_node.meta
        _copy_over_q_dq_args(*node_map['conv_weight_q'])
        _copy_over_q_dq_args(*node_map['conv_weight_dq'])
        if 'conv_bias_q' in node_map:
            assert 'conv_bias_dq' in node_map
            _copy_over_q_dq_args(*node_map['conv_bias_q'])
            _copy_over_q_dq_args(*node_map['conv_bias_dq'])
        conv_bias = None
        (_, conv_node) = node_map['conv']
        (_, bn_node) = node_map['bn']
        (_, conv_weight) = node_map['conv_weight']
        if 'conv_bias' in node_map:
            (_, conv_bias) = node_map['conv_bias']
        fold_bn_weights_into_conv_node(conv_node, conv_weight, conv_bias, bn_node, m)
        for original_node in _filter_nodes_map(r.nodes_map).values():
            if original_node.target == torch.ops.aten.conv2d.default:
                _copy_over_literal_conv_args(original_node, conv_node)
    m.graph.eliminate_dead_code()
    m.recompile()
    return m