import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from .. import config
from ..pattern_matcher import CallFunctionVarArgs, get_arg_value, stable_topological_sort
if config.is_fbcode():
    from torch._inductor.fb.utils import get_everpaste_url
try:
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings
    has_fbgemm = True
except Exception:
    has_fbgemm = False
    pass
aten = torch.ops.aten
log = logging.getLogger(__name__)
MIN_FUSE_SET_SIZE = 5
MAX_FUSE_SET_SIZE = 300
MAX_FUSE_SEARCH_DEPTH = 5
MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR = 4096
SEARCH_EXCLUSIONS = {operator.getitem}

class GroupBatchFusionBase:

    def match(self, node):
        if False:
            print('Hello World!')
        raise NotImplementedError('match called on base')

    def fuse(self, graph, subset):
        if False:
            print('Hello World!')
        raise NotImplementedError('fuse called on base')

class GroupFusion(GroupBatchFusionBase):
    """
    Fuse ops in a group way, e.g, fuse mm/addmm of arbitrary input shapes with fbgemm.gmm.
    """
    pass

class BatchFusion(GroupBatchFusionBase):
    """
    Fuse ops in a batch way, e.g, fuse mm/addmm of same input shapes with bmm.
    """
    pass

class GroupLinearFusion(GroupFusion):

    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        if False:
            while True:
                i = 10
        input_shape = node.args[1].meta['tensor_meta'].shape
        weight_shape = node.args[2].meta['tensor_meta'].shape
        return node.kwargs.get('beta', 1.0) == 1.0 and node.kwargs.get('alpha', 1.0) == 1.0 and (len(input_shape) == 2) and (len(weight_shape) == 2) and all((x % 2 == 0 for x in input_shape + weight_shape)) and all((shape <= MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR for shape in input_shape + weight_shape))

    def _mm_node_can_be_fused(self, node: torch.fx.Node):
        if False:
            while True:
                i = 10
        input_shape = node.args[0].meta['tensor_meta'].shape
        weight_shape = node.args[1].meta['tensor_meta'].shape
        return len(input_shape) == 2 and len(weight_shape) == 2 and all((x % 2 == 0 for x in input_shape + weight_shape)) and all((shape <= MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR for shape in input_shape + weight_shape))

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool]]:
        if False:
            for i in range(10):
                print('nop')
        if CallFunctionVarArgs(aten.mm.default).match(node) and self._mm_node_can_be_fused(node):
            group_key = ('group_linear', True)
        elif CallFunctionVarArgs(aten.addmm.default).match(node) and self._addmm_node_can_be_fused(node):
            bias = node.args[0]
            group_key = ('group_linear', bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        if False:
            for i in range(10):
                print('nop')
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                (bias, input, weight) = node.args
            else:
                assert CallFunctionVarArgs(aten.mm.default).match(node)
                (input, weight) = node.args
                bias = None
            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)
        if all((bias is None for bias in group_biases)):
            group_biases = None
        group_biases: Optional[List[Any]]
        with graph.inserting_before(subset[0]):
            fused_mm = graph.call_function(torch.ops.fbgemm.gmm.default, args=(group_inputs, group_weights, group_biases))
        for (i, original_mm) in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)

class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool, Any]]:
        if False:
            for i in range(10):
                print('nop')
        if CallFunctionVarArgs(torch.nn.functional.linear).match(node) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, 'input')
            bias = get_arg_value(node, 2, 'bias')
            group_key = ('batch_linear_lhs', bias is None, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        if False:
            print('Hello World!')
        batch_nodes = []
        batch_input = None
        batch_weights = []
        batch_biases = []
        split_sections = []
        for node in subset:
            input = get_arg_value(node, 0, 'input')
            weight = get_arg_value(node, 1, 'weight')
            bias = get_arg_value(node, 2, 'bias')
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                assert batch_input is input
            batch_weights.append(weight)
            if bias:
                batch_biases.append(bias)
            split_sections.append(weight.meta['example_value'].shape[0])
        with graph.inserting_before(subset[0]):
            cat_weights = graph.call_function(torch.cat, args=(batch_weights,), kwargs={'dim': 0})
            transposed_weights = graph.call_function(torch.transpose, args=(cat_weights, 0, 1))
            if len(batch_biases) > 0:
                cat_biases = graph.call_function(torch.cat, args=(batch_biases,), kwargs={'dim': 0})
                fused_lhs = graph.call_function(torch.addmm, args=(cat_biases, batch_input, transposed_weights))
            else:
                fused_lhs = graph.call_function(torch.mm, args=(batch_input, transposed_weights))
            fused_lhs_list = graph.call_function(torch.split, args=(fused_lhs, split_sections), kwargs={'dim': 1})
        for (i, node) in enumerate(batch_nodes):
            with graph.inserting_after(fused_lhs_list):
                new_node = graph.call_function(operator.getitem, args=(fused_lhs_list, i))
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)

def is_node_meta_valid(node: Optional[torch.fx.Node]):
    if False:
        print('Hello World!')
    if node is None:
        return True
    if 'example_value' not in node.meta:
        return False
    return True

def is_linear_node_can_be_fused(node: torch.fx.Node):
    if False:
        print('Hello World!')
    input = get_arg_value(node, 0, 'input')
    weight = get_arg_value(node, 1, 'weight')
    return is_node_meta_valid(node) and is_node_meta_valid(input) and is_node_meta_valid(weight) and (len(input.meta['example_value'].shape) == 2) and (len(weight.meta['example_value'].shape) == 2)

class BatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if False:
            return 10
        if getitem_node.target != operator.__getitem__ or getitem_node.op != 'call_function':
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if False:
            print('Hello World!')
        if CallFunctionVarArgs(torch.nn.functional.linear).match(node) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, 'input')
            weight = get_arg_value(node, 1, 'weight')
            bias = get_arg_value(node, 2, 'bias')
            group_key = ('batch_linear_pre_grad', self._getitem_args(input), str(input.meta['example_value'].shape), str(weight.meta['example_value'].shape), bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        if False:
            for i in range(10):
                print('nop')
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        for node in subset:
            batch_nodes.append(node)
            batch_inputs.append(get_arg_value(node, 0, 'input'))
            batch_weights.append(get_arg_value(node, 1, 'weight'))
            batch_biases.append(get_arg_value(node, 2, 'bias'))
        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(torch.stack, args=(batch_inputs,), kwargs={'dim': 0})
            stack_weights = graph.call_function(torch.stack, args=(batch_weights,), kwargs={'dim': 0})
            transpose_weight = graph.call_function(torch.transpose, args=(stack_weights, 1, 2))
            if all((bias is None for bias in batch_biases)):
                bmm = graph.call_function(torch.bmm, args=(stack_inputs, transpose_weight))
            else:
                stack_biases = graph.call_function(torch.stack, args=(batch_biases,), kwargs={'dim': 0})
                unsqueeze_biases = graph.call_function(torch.unsqueeze, args=(stack_biases, 1))
                bmm = graph.call_function(torch.baddbmm, args=(unsqueeze_biases, stack_inputs, transpose_weight))
            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={'dim': 0})
            for (i, linear) in enumerate(batch_nodes):
                with graph.inserting_after(bmm):
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))
                linear.replace_all_uses_with(getitem)
                getitem.meta.update(linear.meta)
                graph.erase_node(linear)

class BatchTanhFusion(BatchFusion):
    """
    Batch tanh fusion in pre grad pass.
    We only fuse the tahn if the input is after same split node.
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if False:
            i = 10
            return i + 15
        if getitem_node.target != operator.__getitem__ or getitem_node.op != 'call_function':
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if False:
            while True:
                i = 10
        input = get_arg_value(node, 0, 'input')
        if CallFunctionVarArgs(torch.tanh).match(node) and is_node_meta_valid(node) and (self._getitem_args(input) is not None):
            group_key = ('batch_tanh', self._getitem_args(input), str(input.meta['example_value'].shape))
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        if False:
            for i in range(10):
                print('nop')
        batch_nodes = []
        batch_inputs = []
        for node in subset:
            batch_nodes.append(node)
            batch_inputs.append(get_arg_value(node, 0, 'input'))
        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(torch.stack, args=(batch_inputs,), kwargs={'dim': 0})
            batch_tanh = graph.call_function(torch.tanh, args=(stack_inputs,))
            unbind_tanh = graph.call_function(torch.unbind, args=(batch_tanh,), kwargs={'dim': 0})
            for (i, node) in enumerate(batch_nodes):
                with graph.inserting_after(unbind_tanh):
                    getitem = graph.call_function(operator.getitem, args=(unbind_tanh, i))
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)

class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """

    def match(self, node: torch.fx.Node):
        if False:
            while True:
                i = 10
        if CallFunctionVarArgs(torch.nn.functional.layer_norm).match(node):
            input = get_arg_value(node, 0, 'input')
            weight = get_arg_value(node, 2, 'weight')
            bias = get_arg_value(node, 3, 'bias')
            group_key = ('batch_layernorm', str(input.meta['example_value'].shape), str(weight.meta['example_value'].shape) if weight is not None else '', str(bias.meta['example_value'].shape) if bias is not None else '', str(get_arg_value(node, 1, 'normalized_shape')), str(get_arg_value(node, 4, 'eps'))) if 'example_value' in input.meta and is_node_meta_valid(weight) and is_node_meta_valid(bias) else None
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        if False:
            while True:
                i = 10
        group_inputs = []
        group_shapes = []
        group_weights = []
        group_biases = []
        group_epss = []
        group_nodes = []
        for node in subset:
            group_nodes.append(node)
            group_inputs.append(get_arg_value(node, 0, 'input'))
            group_shapes.append(get_arg_value(node, 1, 'normalized_shape'))
            group_weights.append(get_arg_value(node, 2, 'weight'))
            group_biases.append(get_arg_value(node, 3, 'bias'))
            eps = get_arg_value(node, 4, 'eps')
            if eps is None:
                eps = 1e-05
            group_epss.append(eps)
        stack_dim = -1 - len(group_shapes[-1])
        if all((bias is None for bias in group_biases)):
            group_biases = None
        group_biases: Optional[List[Any]]
        if all((weight is None for weight in group_weights)):
            group_weights = None
        group_weights: Optional[List[Any]]
        assert all((eps == group_epss[0] for eps in group_epss)), 'all epsilon values must be equal'
        with graph.inserting_before(subset[0]):
            stack_input = graph.call_function(torch.stack, args=(group_inputs,), kwargs={'dim': stack_dim})
            if group_weights is not None:
                stack_weight = graph.call_function(torch.stack, args=(group_weights,), kwargs={'dim': 0})
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(torch.stack, args=(group_biases,), kwargs={'dim': 0})
            else:
                stack_bias = None
            batch_layer_norm = graph.call_function(torch.nn.functional.layer_norm, args=(stack_input, group_shapes[-1]), kwargs={'eps': group_epss[-1]})
            if group_weights is not None and group_biases is not None:
                batch_layer_norm = graph.call_function(torch.addcmul, args=(stack_bias, stack_weight, batch_layer_norm))
            elif group_weights is not None and group_biases is None:
                batch_layer_norm = graph.call_function(torch.mul, args=(stack_weight, batch_layer_norm))
            elif group_weights is None and group_biases is not None:
                batch_layer_norm = graph.call_function(torch.add, args=(stack_bias, batch_layer_norm))
            batch_layer_norm_unbind = graph.call_function(torch.unbind, args=(batch_layer_norm,), kwargs={'dim': stack_dim})
        for (i, node) in enumerate(group_nodes):
            with graph.inserting_after(batch_layer_norm_unbind):
                new_node = graph.call_function(operator.getitem, args=(batch_layer_norm_unbind, i))
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)

class BatchReLUFusion(BatchFusion):
    """
    Batch relu fusion in pre grad pass.
    We only fuse the relu if the input is after same split/unbind node.
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if False:
            i = 10
            return i + 15
        if getitem_node.target != operator.__getitem__ or getitem_node.op != 'call_function':
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if False:
            return 10
        input = get_arg_value(node, 0, 'input')
        if CallFunctionVarArgs(torch.nn.functional.relu).match(node) and is_node_meta_valid(node) and (self._getitem_args(input) is not None):
            group_key = ('batch_relu', self._getitem_args(input), str(input.meta['example_value'].shape))
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        if False:
            print('Hello World!')
        batch_nodes = []
        batch_inputs = []
        for node in subset:
            batch_nodes.append(node)
            batch_inputs.append(get_arg_value(node, 0, 'input'))
        inplace = subset[0].kwargs.get('inplace', False)
        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(torch.stack, args=(batch_inputs,), kwargs={'dim': 0})
            batch_relu = graph.call_function(torch.nn.functional.relu, args=(stack_inputs,), kwargs={'inplace': inplace})
            unbind_relu = graph.call_function(torch.unbind, args=(batch_relu,), kwargs={'dim': 0})
            for (i, node) in enumerate(batch_nodes):
                with graph.inserting_after(unbind_relu):
                    getitem = graph.call_function(operator.getitem, args=(unbind_relu, i))
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)

def find_independent_subset_greedy(node_list: List[torch.fx.Node]) -> Iterator[List[torch.fx.Node]]:
    if False:
        print('Hello World!')
    "\n    Return a list of subset from node_list, all nodes in each subset are independent with each other and can be fused together.\n    The type of subset is list, so we can preserve node's order and benefit from split-cat elimination in later pass.\n    "
    visited_node_set: Set[torch.fx.Node] = set()
    dep_set: Set[torch.fx.Node] = set()

    def find_dependent_nodes(src_node, cur_node):
        if False:
            return 10
        for input_node in cur_node.all_input_nodes:
            if input_node in node_list:
                dep_set.add(input_node)
            if input_node not in visited_node_set:
                visited_node_set.add(input_node)
                find_dependent_nodes(src_node, input_node)
    while len(node_list) > 0:
        subset: List[torch.fx.Node] = []
        subset_deps: Set[torch.fx.Node] = set()
        for node in node_list:
            if len(subset) >= MAX_FUSE_SET_SIZE:
                break
            visited_node_set.clear()
            dep_set.clear()
            find_dependent_nodes(node, node)
            if not dep_set.intersection(subset) and node not in subset_deps:
                subset.append(node)
                subset_deps.update(dep_set)
        if len(subset) >= MIN_FUSE_SET_SIZE:
            yield subset
        next_round_node_list = [node for node in node_list if node not in subset]
        node_list = next_round_node_list

def get_fusion_candidates(rule: GroupBatchFusionBase, root_node: torch.fx.Node, fused_set: Set[torch.fx.Node]) -> DefaultDict[Any, List[torch.fx.Node]]:
    if False:
        return 10
    '\n    Search fusion candidates for a specific rule using BFS starting from the root node.\n    We only search the subgraph within MAX_FUSE_SEARCH_DEPTH.\n    '
    q: Deque[Tuple[int, torch.fx.Node]] = collections.deque()
    candidate_dict: DefaultDict[Any, List[torch.fx.Node]] = collections.defaultdict(list)
    if root_node.target in SEARCH_EXCLUSIONS:
        return candidate_dict
    visited_set: Set[torch.fx.Node] = set()
    for next_node in root_node.all_input_nodes:
        q.append((1, next_node))
        visited_set.add(next_node)
    while len(q) > 0:
        (depth, node) = q.popleft()
        if node in fused_set:
            continue
        key = rule.match(node)
        if key is not None:
            candidate_nodes = candidate_dict[key]
            if node not in candidate_nodes:
                candidate_nodes.append(node)
        elif depth < MAX_FUSE_SEARCH_DEPTH:
            for next_node in node.all_input_nodes:
                if next_node not in visited_set:
                    visited_set.add(next_node)
                    q.append((depth + 1, next_node))
    return candidate_dict

def apply_group_batch_fusion(graph: torch.fx.GraphModule, rule: GroupBatchFusionBase):
    if False:
        print('Hello World!')
    stable_topological_sort(graph)
    fused_set: Set[torch.fx.Node] = set()
    for node in reversed(graph.nodes):
        candidates = get_fusion_candidates(rule, node, fused_set)
        for (key, candidate_nodes) in candidates.items():
            if len(candidate_nodes) < MIN_FUSE_SET_SIZE:
                continue
            for subset in find_independent_subset_greedy(candidate_nodes):
                rule.fuse(graph, subset)
                fused_set.update(subset)
                if isinstance(rule, GroupFusion):
                    counters['inductor']['group_fusion'] += 1
                else:
                    counters['inductor']['batch_fusion'] += 1
                log.info(f'{rule.__class__.__name__}: key = {key}; subset size = {len(subset)}')

def print_graph(graph: torch.fx.Graph, msg: str):
    if False:
        print('Hello World!')
    if config.is_fbcode():
        log.info('%s Print graph: %s', msg, get_everpaste_url(str(graph)))

def group_batch_fusion_post_grad_passes(graph: torch.fx.Graph):
    if False:
        return 10
    print_graph(graph, 'Before group_batch fusion in post grads pass.')
    fusions: List[GroupBatchFusionBase] = []
    if config.group_fusion and has_fbgemm:
        fusions += [GroupLinearFusion()]
    for rule in fusions:
        apply_group_batch_fusion(graph, rule)
        print_graph(graph, f'Apply fusion {rule.__class__.__name__}.')

def group_batch_fusion_pre_grad_passes(graph: torch.fx.Graph):
    if False:
        print('Hello World!')
    print_graph(graph, 'Before group_batch fusion in pre grads pass.')
    fusions: List[GroupBatchFusionBase] = []
    if config.batch_fusion:
        fusions += [BatchLinearFusion(), BatchLinearLHSFusion(), BatchLayernormFusion(), BatchTanhFusion(), BatchReLUFusion()]
    for rule in fusions:
        apply_group_batch_fusion(graph, rule)
        print_graph(graph, f'Apply fusion {rule.__class__.__name__}.')