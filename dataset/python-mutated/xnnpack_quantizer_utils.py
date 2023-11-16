import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer import QuantizationAnnotation, QuantizationSpec, QuantizationSpecBase, SharedQuantizationSpec
from torch.ao.quantization.quantizer.utils import _annotate_input_qspec_map, _annotate_output_qspec
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
__all__ = ['OperatorConfig', 'OperatorPatternType', 'QuantizationConfig', 'get_input_act_qspec', 'get_output_act_qspec', 'get_weight_qspec', 'get_bias_qspec', 'OP_TO_ANNOTATOR', 'propagate_annotation']

@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
    is_qat: bool = False
OperatorPatternType = List[Callable]
OperatorPatternType.__module__ = 'torch.ao.quantization.quantizer.xnnpack_quantizer_utils'
AnnotatorType = Callable[[torch.fx.GraphModule, Optional[QuantizationConfig], Optional[Callable[[Node], bool]]], Optional[List[List[Node]]]]
OP_TO_ANNOTATOR: Dict[str, AnnotatorType] = {}

def register_annotator(op: str):
    if False:
        return 10

    def decorator(annotator: AnnotatorType):
        if False:
            i = 10
            return i + 15
        OP_TO_ANNOTATOR[op] = annotator
    return decorator

class OperatorConfig(NamedTuple):
    config: QuantizationConfig
    operators: List[OperatorPatternType]

def _is_annotated(nodes: List[Node]):
    if False:
        print('Hello World!')
    '\n    Given a list of nodes (that represents an operator pattern),\n    check if any of the node is annotated, return True if any of the node\n    is annotated, otherwise return False\n    '
    annotated = False
    for node in nodes:
        annotated = annotated or ('quantization_annotation' in node.meta and node.meta['quantization_annotation']._annotated)
    return annotated

def _mark_nodes_as_annotated(nodes: List[Node]):
    if False:
        print('Hello World!')
    for node in nodes:
        if node is not None:
            if 'quantization_annotation' not in node.meta:
                node.meta['quantization_annotation'] = QuantizationAnnotation()
            node.meta['quantization_annotation']._annotated = True

def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if False:
        for i in range(10):
            print('nop')
    if quantization_config is None:
        return None
    if quantization_config.input_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.input_activation
    assert quantization_spec.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]
    return quantization_spec

def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if False:
        i = 10
        return i + 15
    if quantization_config is None:
        return None
    if quantization_config.output_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.output_activation
    assert quantization_spec.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]
    return quantization_spec

def get_weight_qspec(quantization_config: Optional[QuantizationConfig]):
    if False:
        return 10
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight
    if quantization_spec.qscheme not in [torch.per_tensor_symmetric, torch.per_channel_symmetric]:
        raise ValueError(f'Unsupported quantization_spec {quantization_spec} for weight')
    return quantization_spec

def get_bias_qspec(quantization_config: Optional[QuantizationConfig]):
    if False:
        for i in range(10):
            print('nop')
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert quantization_spec.dtype == torch.float, 'Only float dtype for bias is supported for bias right now'
    return quantization_spec

@register_annotator('linear')
def _annotate_linear(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        return 10
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != 'call_function' or node.target != torch.ops.aten.linear.default:
            continue
        if filter_fn and (not filter_fn(node)):
            continue
        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[2]
        if _is_annotated([node]) is False:
            _annotate_input_qspec_map(node, act_node, input_act_qspec)
            _annotate_input_qspec_map(node, weight_node, weight_qspec)
            nodes_to_mark_annotated = [node, weight_node]
            if bias_node:
                _annotate_input_qspec_map(node, bias_node, bias_qspec)
                nodes_to_mark_annotated.append(bias_node)
            _annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)
    return annotated_partitions

@register_annotator('conv')
def _annotate_conv(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        i = 10
        return i + 15
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != 'call_function' or n.target not in [torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default]:
            continue
        conv_node = n
        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)
        partition = [conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)
        if _is_annotated(partition):
            continue
        if filter_fn and any((not filter_fn(n) for n in partition)):
            continue
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions

@register_annotator('conv_relu')
def _annotate_conv_relu(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        print('Hello World!')
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != 'call_function' or n.target not in [torch.ops.aten.relu.default, torch.ops.aten.relu_.default]:
            continue
        relu_node = n
        maybe_conv_node = n.args[0]
        if not isinstance(maybe_conv_node, Node) or maybe_conv_node.op != 'call_function' or maybe_conv_node.target not in [torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default]:
            continue
        conv_node = maybe_conv_node
        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)
        partition = [relu_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)
        if _is_annotated(partition):
            continue
        if filter_fn and any((not filter_fn(n) for n in partition)):
            continue
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        relu_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions

@register_annotator('conv2d_bn')
def _annotate_conv2d_bn(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Find Conv2d + batchnorm parititions\n    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.\n    '
    fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d], filter_fn)
    annotated_partitions = []
    for fused_partition in fused_partitions:
        (conv_partition, bn_partition) = fused_partition
        annotated_partitions.append(conv_partition.nodes + bn_partition.nodes)
        if len(conv_partition.output_nodes) > 1:
            raise ValueError('conv partition has more than one output node')
        conv_node = conv_partition.output_nodes[0]
        conv_node_users = list(conv_node.users.keys())
        if len(conv_node_users) > 1:
            raise ValueError('Conv node must be consumed by BN only for it to be fusable.')
        if len(bn_partition.output_nodes) > 1:
            raise ValueError('BatchNorm partition has more than one output node')
        bn_output_node = bn_partition.output_nodes[0]
        if _is_annotated([bn_output_node, conv_node]):
            continue
        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        bn_output_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        nodes_to_mark_annotated = list(conv_partition.nodes)
        nodes_to_mark_annotated.extend(list(bn_partition.nodes))
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
    return annotated_partitions

@register_annotator('conv2d_bn_relu')
def _annotate_conv2d_bn_relu(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Find Conv2d + batchnorm + relu parititions\n    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.\n    '
    fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU], filter_fn)
    annotated_partitions = []
    for fused_partition in fused_partitions:
        (conv_partition, bn_partition, relu_partition) = fused_partition
        annotated_partitions.append(conv_partition.nodes + bn_partition.nodes + relu_partition.nodes)
        if len(relu_partition.output_nodes) > 1:
            raise ValueError('Relu partition has more than one output node')
        relu_node = relu_partition.output_nodes[0]
        if len(conv_partition.output_nodes) > 1:
            raise ValueError('conv partition has more than one output node')
        conv_node = conv_partition.output_nodes[0]
        conv_node_users = list(conv_node.users.keys())
        if len(conv_node_users) > 1:
            raise ValueError('Conv node must be consumed by BN only for it to be fusable.')
        if len(bn_partition.output_nodes) > 1:
            raise ValueError('BatchNorm partition has more than one output node')
        bn_output_node = bn_partition.output_nodes[0]
        if _is_annotated([relu_node, bn_output_node, conv_node]):
            continue
        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        relu_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        nodes_to_mark_annotated = list(conv_partition.nodes)
        nodes_to_mark_annotated.extend(list(bn_partition.nodes))
        nodes_to_mark_annotated.extend(list(relu_partition.nodes))
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
    return annotated_partitions

@register_annotator('gru_io_only')
def _annotate_gru_io_only(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        while True:
            i = 10
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    gru_partitions = list(itertools.chain(*gru_partitions.values()))
    annotated_partitions = []
    for gru_partition in gru_partitions:
        annotated_partitions.append(gru_partition.nodes)
        output_nodes = gru_partition.output_nodes
        input_nodes = gru_partition.input_nodes
        if _is_annotated(input_nodes + output_nodes):
            continue
        input_qspec_map: Dict[Node, QuantizationSpecBase] = {}
        input_act = input_nodes[0]
        input_act_user = next(iter(input_act.users.keys()))
        assert isinstance(input_act, Node)
        assert isinstance(input_act_user, Node)
        input_act_user.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: get_input_act_qspec(quantization_config)}, _annotated=True)
        hidden_state = input_nodes[1]
        hidden_state_user = next(iter(hidden_state.users.keys()))
        assert isinstance(hidden_state, Node)
        assert isinstance(hidden_state_user, Node)
        hidden_state_user.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={hidden_state: get_input_act_qspec(quantization_config)}, _annotated=True)
        assert len(output_nodes) == 2, 'expecting GRU to have two outputs'
        for output in output_nodes:
            output.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        nodes_to_mark_annotated = list(gru_partition.nodes)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
    return annotated_partitions

@register_annotator('max_pool2d')
def _annotate_max_pool2d(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        i = 10
        return i + 15
    module_partitions = get_source_partitions(gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d], filter_fn)
    maxpool_partitions = list(itertools.chain(*module_partitions.values()))
    annotated_partitions = []
    for maxpool_partition in maxpool_partitions:
        annotated_partitions.append(maxpool_partition.nodes)
        output_node = maxpool_partition.output_nodes[0]
        maxpool_node = None
        for n in maxpool_partition.nodes:
            if n.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = n
        assert maxpool_node is not None, 'XNNPACKQuantizer only works with torch.ops.aten.max_pool2d.default, '
        'please make sure you are exporting the model correctly'
        if _is_annotated([output_node, maxpool_node]):
            continue
        input_act = maxpool_node.args[0]
        assert isinstance(input_act, Node)
        if 'quantization_annotation' not in input_act.meta or not input_act.meta['quantization_annotation']._annotated or input_act.meta['quantization_annotation'].output_qspec is None:
            continue
        act_qspec = SharedQuantizationSpec(input_act)
        maxpool_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec}, _annotated=True)
        output_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=act_qspec, _annotated=True)
    return annotated_partitions

@register_annotator('adaptive_avg_pool2d')
def _annotate_adaptive_avg_pool2d(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        return 10
    'Always annotate adaptive_avg_pool2d op'
    module_partitions = get_source_partitions(gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], filter_fn)
    partitions = list(itertools.chain(*module_partitions.values()))
    annotated_partitions = []
    for partition in partitions:
        pool_node = partition.output_nodes[0]
        if pool_node.op != 'call_function' or pool_node.target != torch.ops.aten.adaptive_avg_pool2d.default:
            raise ValueError(f'{pool_node} is not an aten adaptive_avg_pool2d operator')
        if _is_annotated([pool_node]):
            continue
        annotated_partitions.append(partition.nodes)
        input_act = pool_node.args[0]
        assert isinstance(input_act, Node)
        if 'quantization_annotation' not in input_act.meta or not input_act.meta['quantization_annotation']._annotated or input_act.meta['quantization_annotation'].output_qspec is None:
            input_act_qspec = get_input_act_qspec(quantization_config)
        else:
            input_act_qspec = SharedQuantizationSpec(input_act)
        output_act_qspec = SharedQuantizationSpec((input_act, pool_node))
        pool_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: input_act_qspec}, output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions

@register_annotator('add_relu')
def _annotate_add_relu(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        print('Hello World!')
    fused_partitions = find_sequential_partitions(gm, [torch.add, torch.nn.ReLU], filter_fn)
    annotated_partitions = []
    for fused_partition in fused_partitions:
        (add_partition, relu_partition) = fused_partition
        annotated_partitions.append(add_partition.nodes + relu_partition.nodes)
        if len(relu_partition.output_nodes) > 1:
            raise ValueError('Relu partition has more than one output node')
        relu_node = relu_partition.output_nodes[0]
        if len(add_partition.output_nodes) > 1:
            raise ValueError('add partition has more than one output node')
        add_node = add_partition.output_nodes[0]
        if _is_annotated([relu_node, add_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec
        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec
        add_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        relu_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions

@register_annotator('add')
def _annotate_add(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        for i in range(10):
            print('nop')
    add_partitions = get_source_partitions(gm.graph, [operator.add, torch.add, operator.iadd], filter_fn)
    add_partitions = list(itertools.chain(*add_partitions.values()))
    annotated_partitions = []
    for add_partition in add_partitions:
        annotated_partitions.append(add_partition.nodes)
        add_node = add_partition.output_nodes[0]
        if _is_annotated([add_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec
        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec
        add_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions

@register_annotator('mul_relu')
def _annotate_mul_relu(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        i = 10
        return i + 15
    fused_partitions = find_sequential_partitions(gm, [torch.mul, torch.nn.ReLU], filter_fn)
    annotated_partitions = []
    for fused_partition in fused_partitions:
        (mul_partition, relu_partition) = fused_partition
        annotated_partitions.append(mul_partition.nodes + relu_partition.nodes)
        if len(relu_partition.output_nodes) > 1:
            raise ValueError('Relu partition has more than one output node')
        relu_node = relu_partition.output_nodes[0]
        if len(mul_partition.output_nodes) > 1:
            raise ValueError('mul partition has more than one output node')
        mul_node = mul_partition.output_nodes[0]
        if _is_annotated([relu_node, mul_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec
        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec
        mul_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        relu_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions

@register_annotator('mul')
def _annotate_mul(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        i = 10
        return i + 15
    mul_partitions = get_source_partitions(gm.graph, ['mul', 'mul_', operator.mul, torch.mul, operator.imul], filter_fn)
    mul_partitions = list(itertools.chain(*mul_partitions.values()))
    annotated_partitions = []
    for mul_partition in mul_partitions:
        annotated_partitions.append(mul_partition.nodes)
        mul_node = mul_partition.output_nodes[0]
        if _is_annotated([mul_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec
        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec
        mul_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions

@register_annotator('cat')
def _annotate_cat(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    if False:
        while True:
            i = 10
    cat_partitions = get_source_partitions(gm.graph, [torch.cat], filter_fn)
    cat_partitions = list(itertools.chain(*cat_partitions.values()))
    annotated_partitions = []
    for cat_partition in cat_partitions:
        cat_node = cat_partition.output_nodes[0]
        if _is_annotated([cat_node]):
            continue
        if cat_node.target != torch.ops.aten.cat.default:
            raise Exception(f'Expected cat node: torch.ops.aten.cat.default, but found {cat_node.target} please check if you are calling the correct capture API')
        annotated_partitions.append(cat_partition.nodes)
        input_act_qspec = get_input_act_qspec(quantization_config)
        inputs = cat_node.args[0]
        input_qspec_map = {}
        input_act0 = inputs[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec
        shared_with_input0_qspec = SharedQuantizationSpec((input_act0, cat_node))
        for input_act in inputs[1:]:
            input_qspec_map[input_act] = shared_with_input0_qspec
        output_act_qspec = shared_with_input0_qspec
        cat_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions

def _is_share_obs_or_fq_op(op: Callable) -> bool:
    if False:
        while True:
            i = 10
    return op in [torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default, torch.ops.aten.mean.default, torch.ops.aten.mean.dim, torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default, torch.ops.aten.squeeze.dim, torch.ops.aten.squeeze_copy.dim, torch.ops.aten.adaptive_avg_pool2d.default, torch.ops.aten.view_copy.default, torch.ops.aten.view.default, torch.ops.aten.slice_copy.Tensor, torch.ops.aten.flatten.using_ints]

def propagate_annotation(model: torch.fx.GraphModule) -> None:
    if False:
        return 10
    for n in model.graph.nodes:
        if n.op != 'call_function' or not _is_share_obs_or_fq_op(n.target):
            continue
        prev_node = n.args[0]
        if not isinstance(prev_node, Node):
            continue
        quantization_annotation = prev_node.meta.get('quantization_annotation', None)
        if not quantization_annotation:
            continue
        output_qspec = quantization_annotation.output_qspec
        if not output_qspec:
            continue
        if 'quantization_annotation' in n.meta and n.meta['quantization_annotation']._annotated:
            continue
        shared_qspec = SharedQuantizationSpec(prev_node)
        n.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={prev_node: shared_qspec}, output_qspec=shared_qspec, _annotated=True)

def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    if False:
        while True:
            i = 10
    for n in model.graph.nodes:
        if n.op != 'call_function' or n.target not in [torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor]:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = '_tensor_constant_'
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            model.register_buffer(tensor_constant_name, torch.tensor(float(args[i])))
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node('get_attr', tensor_constant_name, (), {})
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()
    return model