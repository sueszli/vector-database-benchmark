import torch
import torch.fx
from torch.fx import Node, GraphModule, Graph
from torch.ao.ns.fx.utils import get_target_type_str, get_normalized_nth_input
from torch.ao.ns.fx.ns_types import NSSingleResultValuesType, NSResultsType
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
SHADOW_NODE_NAME_PREFIX = 'shadow'
SHADOW_WRAPPER_NODE_NAME_PREFIX = 'shadow_wrapper'
BINARY_FUNCTIONS = {torch.add, torch.Tensor.add, operator.add, torch.mul, torch.Tensor.mul, operator.mul}

def _get_attr_name(subgraph_idx, subgraph_candidate_idx):
    if False:
        print('Hello World!')
    return f'{SHADOW_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}'

def _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx):
    if False:
        while True:
            i = 10
    return f'{SHADOW_WRAPPER_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}'

class OutputProp:
    """
    Output propagation (modeled from shape propagation).

    Given a GraphModule and an example input, saves the output flowing
    through each node on `node.traced_result`.

    Code based on the example from
    https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
    """

    def __init__(self, mod):
        if False:
            i = 10
            return i + 15
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        if False:
            return 10
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            if False:
                print('Hello World!')
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            if False:
                for i in range(10):
                    print('nop')
            target_atoms = target.split('.')
            attr_itr = self.mod
            for (i, atom) in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                (self_obj, *args) = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            if isinstance(result, torch.Tensor):
                node.traced_result = result
            env[node.name] = result
        return None

def _get_dedup_subgraphs(matches: Dict[str, _MatchResult]) -> Dict[str, List[Node]]:
    if False:
        print('Hello World!')
    seen_nodes = set()
    subgraphs_dedup = {}
    matches_items_reversed: List[Tuple[str, _MatchResult]] = []
    for (name, cur_match) in matches.items():
        matches_items_reversed.insert(0, (name, cur_match))
    for (name, cur_match) in matches_items_reversed:
        was_seen = False
        for node_or_tuple in cur_match[1]:
            if isinstance(node_or_tuple, Node):
                if node_or_tuple in seen_nodes:
                    was_seen = True
                seen_nodes.add(node_or_tuple)
            else:
                assert isinstance(node_or_tuple, tuple)
                for node in node_or_tuple:
                    assert isinstance(node, Node)
                    if node in seen_nodes:
                        was_seen = True
                    seen_nodes.add(node)
        if was_seen:
            continue
        list_of_nodes = []
        if len(cur_match[1]) == 1:
            list_of_nodes = cur_match[1]
        else:
            assert len(cur_match[1]) == 2

            def _order_nodes(node_a, node_b, node_c) -> List[Node]:
                if False:
                    for i in range(10):
                        print('nop')
                nodes = [node_a, node_b, node_c]
                first_node = None
                mid_node = None
                last_node = None
                for n in nodes:
                    prev_n = n.args[0]
                    next_n = next(iter(n.users))
                    if prev_n not in nodes:
                        first_node = n
                    elif next_n not in nodes:
                        last_node = n
                    else:
                        mid_node = n
                assert first_node is not None and mid_node is not None and (last_node is not None)
                assert mid_node.args[0] is first_node
                assert last_node.args[0] is mid_node
                return [last_node, mid_node, first_node]
            if isinstance(cur_match[1][0], Node) and isinstance(cur_match[1][1], Node):
                list_of_nodes = cur_match[1]
            elif isinstance(cur_match[1][0], tuple):
                (node_a, node_b) = cur_match[1][0]
                node_c = cur_match[1][1]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)
            elif isinstance(cur_match[1][1], tuple):
                (node_a, node_b) = cur_match[1][1]
                node_c = cur_match[1][0]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)
        list_of_nodes.reverse()
        subgraphs_dedup[name] = list_of_nodes
    return subgraphs_dedup

def _get_logger_for_subgraph(model: GraphModule, first_node: Node, last_node: Node, subgraph_idx: int, subgraph_candidate_idx: int, qconfig_str: str, logger_cls: Callable, fqn: Optional[str]) -> torch.nn.Module:
    if False:
        return 10
    '\n    Given a model and a linear subgraph starting from `first_node` and\n    ending with `last_node`, creates a logger for the end of this\n    subgraph.\n    '
    if fqn is None:
        fqn = ''
    logger_mod_orig = logger_cls(first_node.name, last_node.name, f'subgraph_{subgraph_idx}_{subgraph_candidate_idx}', 'model', get_target_type_str(last_node, model), get_target_type_str(first_node, model), NSSingleResultValuesType.NODE_OUTPUT.value, 0, 0, fqn, qconfig_str)
    logger_mod_orig.enabled = False
    return logger_mod_orig

def create_submodule_from_subgraph(model: torch.nn.Module, first_node: Node, last_node: Node) -> GraphModule:
    if False:
        i = 10
        return i + 15
    '\n    Input: a model, and a linear subgraph within the model from first_node to\n      last_node.\n\n    Output: a new submodule containing a copy of the subgraph, with the inputs\n      to the first node becoming the inputs to the submodule, and all other\n      nodes in the subgraph being copied.\n\n    Example inputs:\n\n    `model`: a module with graph\n\n      x0 -> op1 -> x1 -> op2 -> x2\n             |\n            arg1\n\n    `first_node`: op1\n    `last_node`: op2\n\n    Example output: a new module with graph\n\n      input1 -> op1_copy -> x1 -> op2_copy -> output1\n                   |\n                  arg1\n    '

    class M(torch.nn.Module):

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            pass
    m = M()
    gm = torch.fx.symbolic_trace(m)
    g = gm.graph
    for node in reversed(gm.graph.nodes):
        g.erase_node(node)
    cur_node_orig = first_node
    cur_args_orig = cur_node_orig.args
    cur_kwargs_orig = cur_node_orig.kwargs
    cur_name_idx = 0
    iteration_limit = 100
    cur_iteration = 0
    while True:
        if cur_node_orig is first_node:
            cur_args_copy = []
            cur_kwargs_copy = {}
            seen_names: Set[str] = set()
            old_name_to_new_node: Dict[str, Node] = {}

            def _add_placeholder(g: Graph, node: Node, seen_names, old_name_to_new_node):
                if False:
                    return 10
                counter = 0
                while node.name + '_' + str(counter) in seen_names:
                    counter += 1
                cur_name = node.name + '_' + str(counter)
                seen_names.add(cur_name)
                placeholder = g.placeholder(cur_name)
                old_name_to_new_node[node.name] = placeholder
                return placeholder
            for arg in cur_node_orig.args:
                if isinstance(arg, Node):
                    p = _add_placeholder(g, arg, seen_names, old_name_to_new_node)
                    cur_args_copy.append(p)
                elif isinstance(arg, (list, tuple)):
                    new_arg = []
                    for inner_arg in arg:
                        if isinstance(inner_arg, Node):
                            new_arg.append(_add_placeholder(g, inner_arg, seen_names, old_name_to_new_node))
                        else:
                            new_arg.append(inner_arg)
                    cur_args_copy.append(new_arg)
                else:
                    cur_args_copy.append(arg)
            for (kwarg_name, kwarg) in cur_node_orig.kwargs.items():
                if isinstance(kwarg, Node):
                    cur_kwargs_copy[kwarg_name] = _add_placeholder(g, kwarg, seen_names, old_name_to_new_node)
                elif isinstance(kwarg, (list, tuple)):
                    new_kwarg = []
                    for inner_kwarg in kwarg:
                        p = _add_placeholder(g, inner_kwarg, seen_names, old_name_to_new_node)
                        new_kwarg.append(p)
                    cur_kwargs_copy[kwarg_name] = new_kwarg
                else:
                    cur_kwargs_copy[kwarg_name] = kwarg
            cur_args_copy = tuple(cur_args_copy)
        else:
            assert cur_node_orig.target not in BINARY_FUNCTIONS
            cur_args_copy = [cur_node_copy]
            if len(cur_node_orig.args) > 1:
                for arg in cur_node_orig.args[1:]:
                    if isinstance(arg, torch.nn.Parameter):
                        new_arg = arg.clone().detach()
                        mod_name = f'mod_{cur_name_idx}'
                        cur_name_idx += 1
                        setattr(gm, mod_name, new_arg)
                        new_arg_placeholder = gm.placeholder(mod_name)
                        cur_args_copy.append(new_arg_placeholder)
                    elif isinstance(arg, (float, int, torch.dtype)):
                        cur_args_copy.append(arg)
                    else:
                        raise AssertionError(f'arg of type {type(arg)} not handled yet')
            cur_args_copy = tuple(cur_args_copy)
        if cur_node_orig.op == 'call_module':
            orig_mod = getattr_from_fqn(model, cur_node_orig.target)
            orig_mod_copy = copy.deepcopy(orig_mod)
            mod_name = f'mod_{cur_name_idx}'
            setattr(gm, mod_name, orig_mod_copy)
            cur_name_idx += 1
            cur_node_copy = g.call_module(mod_name, cur_args_copy, cur_kwargs_copy)
        elif cur_node_orig.op == 'call_function':
            cur_node_copy = g.call_function(cur_node_orig.target, cur_args_copy, cur_kwargs_copy)
        elif cur_node_orig.op == 'call_method':
            cur_node_copy = g.call_method(cur_node_orig.target, cur_args_copy, cur_kwargs_copy)
        else:
            raise AssertionError(f'{cur_node_orig.op} not supported yet')
        if cur_node_orig is last_node:
            break
        assert len(cur_node_orig.users.keys()) == 1, f'{cur_node_orig} has more than 1 users, not supported yet'
        cur_node_orig = next(iter(cur_node_orig.users.keys()))
        cur_args_orig = cur_node_orig.args
        cur_kwargs_orig = cur_node_orig.kwargs
        cur_iteration += 1
        if cur_iteration > iteration_limit:
            raise AssertionError('iteration limit exceeded')
    g.output(cur_node_copy)
    gm.recompile()
    return gm

def create_one_transformed_and_logged_copy_of_subgraph(mt: GraphModule, subgraph_idx: int, subgraph_candidate_idx: int, first_node: Node, last_node: Node, fqn: Optional[str], list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]], example_inputs: Any, last_added_shadow_node_list: List[Optional[Node]], custom_prepare_fn: Optional[Callable]=None, custom_prepare_kwargs: Optional[Dict[str, Any]]=None) -> None:
    if False:
        return 10
    '\n    Given a subgraph in `mt` and a subgraph candidate idx, inserts the\n    subgraph candidate copy and instruments it with loggers.\n\n    If subgraph_candidate_idx is 0, this is the baseline fp32 subgraph and we just\n    add a logger to the end.\n\n    If subgraph_candidate_idx is not 0, we create a copy of the subgraph and\n    prepare it with `prepare_fx`.\n    '
    from torch.ao.ns._numeric_suite_fx import OutputLogger, OutputComparisonLogger
    if subgraph_candidate_idx == 0:
        qconfig_str = ''
        logger_mod_orig = _get_logger_for_subgraph(mt, first_node, last_node, subgraph_idx, subgraph_candidate_idx, qconfig_str, OutputLogger, fqn)
        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        assert not hasattr(mt, attr_name)
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(last_node):
            new_node = mt.graph.call_module(attr_name, args=(last_node,), kwargs={})
            last_added_shadow_node_list[0] = new_node
    else:
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]
        if qconfig is None:
            return
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        orig_mod_copy_wrapped = create_submodule_from_subgraph(mt, first_node, last_node)
        if custom_prepare_fn is None:
            orig_mod_copy_wrapped = torch.ao.quantization.quantize_fx.prepare_fx(orig_mod_copy_wrapped, qconfig_mapping, example_inputs=example_inputs)
        else:
            if custom_prepare_kwargs is None:
                custom_prepare_kwargs = {}
            for kwarg_name in ['example_inputs', 'prepare_custom_config', 'qconfig_mapping']:
                assert kwarg_name not in custom_prepare_kwargs, f'cannot specify {kwarg_name} in custom_prepare_kwargs'
            prepare_kwargs: Dict[str, Any] = {'example_inputs': example_inputs, 'qconfig_mapping': qconfig_mapping}
            prepare_kwargs.update(custom_prepare_kwargs)
            orig_mod_copy_wrapped = custom_prepare_fn(orig_mod_copy_wrapped, **prepare_kwargs)
        attr_name = _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx)
        assert not hasattr(mt, attr_name)
        setattr(mt, attr_name, orig_mod_copy_wrapped)
        insert_after_node = last_added_shadow_node_list[0]
        with mt.graph.inserting_after(insert_after_node):
            new_args = []
            for arg in first_node.args:
                if isinstance(arg, Node):
                    new_args.append(arg)
                elif isinstance(arg, (list, tuple)) and len(arg) and isinstance(arg[0], Node):
                    for inner_arg in arg:
                        if isinstance(inner_arg, Node):
                            new_args.append(inner_arg)
            new_kwargs = {}
            for (name, old_kwarg) in first_node.kwargs.items():
                if isinstance(old_kwarg, Node):
                    new_kwargs[name] = old_kwarg
                elif isinstance(old_kwarg, (list, tuple)) and len(old_kwarg):
                    for inner_old_kwarg in old_kwarg:
                        new_args.append(inner_old_kwarg)
            new_args = tuple(new_args)
            new_node = mt.graph.call_module(attr_name, args=new_args, kwargs=new_kwargs)
        logger_mod_orig = _get_logger_for_subgraph(mt, first_node, last_node, subgraph_idx, subgraph_candidate_idx, str(qconfig), OutputComparisonLogger, fqn)
        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        assert not hasattr(mt, attr_name)
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(new_node):
            logger = mt.graph.call_module(attr_name, args=(new_node, last_node), kwargs={})
            last_added_shadow_node_list[0] = logger
    mt.recompile()

def create_n_transformed_and_logged_copies_of_subgraph(mt: GraphModule, subgraph_idx: int, match_name: str, nodes_in_this_subgraph: List[Any], qconfig_mappings: List[QConfigMapping], list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]], custom_prepare_fn: Optional[Callable]=None, custom_prepare_kwargs: Optional[Dict[str, Any]]=None) -> None:
    if False:
        while True:
            i = 10
    '\n    Given a model `mt` and a subgraph_idx, creates the needed copies\n    of the subgraph for all qconfigs, and instruments them with loggers.\n    '
    if any((not isinstance(node, Node) for node in nodes_in_this_subgraph)):
        return
    first_node = nodes_in_this_subgraph[0]
    last_node = nodes_in_this_subgraph[-1]
    prev_node = get_normalized_nth_input(first_node, mt, 0)
    if isinstance(prev_node, list):
        example_inputs = [x.traced_result for x in prev_node]
    elif isinstance(prev_node, tuple):
        example_inputs = (x.traced_result for x in prev_node)
    elif hasattr(prev_node, 'traced_result'):
        example_inputs = (prev_node.traced_result,)
    else:
        print('unable to get example input for node ' + f'{first_node.format_node()}, skipping')
        return
    found_at_least_one_qconfig = False
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        if subgraph_candidate_idx == 0:
            continue
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]
        if qconfig is not None:
            found_at_least_one_qconfig = True
            break
    if not found_at_least_one_qconfig:
        print('unable to find at least one qconfig for node ' + f'{first_node.format_node()}, skipping')
        return
    fqn = _maybe_get_fqn(first_node, mt)
    last_added_shadow_node_list: List[Optional[Node]] = [None]
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        create_one_transformed_and_logged_copy_of_subgraph(mt, subgraph_idx, subgraph_candidate_idx, first_node, last_node, fqn, list_of_node_name_to_qconfig, example_inputs, last_added_shadow_node_list, custom_prepare_fn, custom_prepare_kwargs)

def create_add_loggers_graph(model: GraphModule, subgraphs_dedup: Dict[str, List[Node]], qconfig_mapping: QConfigMapping, node_name_to_qconfig: Dict[str, QConfigAny]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a model, a model graph partition (currently a set of matched\n    subgraphs) and instructions how to transform each subgraph\n    (currently quantizing it according to qconfig_mapping), modifies\n    the model graph to create an alternate path through the original graph,\n    with each of the subgraphs quantized.  This is useful to compare\n    propagation error of a transformation such as quantization.\n\n    For example, given layer op0 and op1, there are four cases when handling op1:\n    1. op0 and op1 quantized\n    2. op0 and op1 unquantized\n    3. op0 quantized, op1 unquantized\n    4. op0 unquantized, op1 quantized\n\n    Example input, case 1:\n\n    .. code::\n\n      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log\n       \\                        \\          \\                 \\       # noqa: W605\n         ---> op0_1 -> x1_1 ----> clog    op1_1 -> x2_1 ----> clog\n\n    Example output, case 1:\n\n    .. code::\n\n      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log\n       \\                        \\                           \\        # noqa: W605\n         ---> op0_1 -> x1_1 ----> clog -> op1_1 -> x2_1 ----> clog\n\n    '
    from torch.ao.ns._numeric_suite_fx import OutputLogger, OutputComparisonLogger

    def _get_subgraph_containing_node(node, subgraphs_dedup):
        if False:
            i = 10
            return i + 15
        for subgraph in subgraphs_dedup.values():
            if node in subgraph:
                return subgraph
        return None
    nodes_to_skip = set()
    orig_first_node_to_shadow_in_node = {}
    orig_first_node_to_shadow_out_node = {}
    orig_nodes = list(model.graph.nodes)
    cur_subgraph_idx = 0
    for n in orig_nodes:
        if n.op in ('placeholder', 'get_attr', 'output') or n in nodes_to_skip:
            continue
        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        insert_submodule_copy = False
        if maybe_subgraph is not None:
            (first_node, last_node) = (maybe_subgraph[0], maybe_subgraph[-1])
            for node_to_skip in maybe_subgraph:
                nodes_to_skip.add(node_to_skip)
            qconfig = node_name_to_qconfig[first_node.name]
            if qconfig is not None:
                insert_submodule_copy = True
        else:
            (first_node, last_node) = (n, n)
        if insert_submodule_copy:
            match_name = first_node.name
            create_n_transformed_and_logged_copies_of_subgraph(model, cur_subgraph_idx, match_name, maybe_subgraph, [qconfig_mapping], [node_name_to_qconfig], None, None)
            expected_shadow_target = f'shadow_wrapper_{cur_subgraph_idx}_1'
            new_shadow_mod = None
            for maybe_shadow_mod in model.graph.nodes:
                if maybe_shadow_mod.op == 'call_module' and maybe_shadow_mod.target == expected_shadow_target:
                    new_shadow_mod = maybe_shadow_mod
                    break
            assert new_shadow_mod is not None
            orig_first_node_to_shadow_in_node[first_node] = new_shadow_mod
            orig_first_node_to_shadow_out_node[first_node] = new_shadow_mod
        else:
            subgraph_to_use = maybe_subgraph if maybe_subgraph is not None else [first_node]
            qconfig_str = ''
            subgraph_candidate_idx = 0
            fqn = _maybe_get_fqn(first_node, model)
            logger_mod_orig = _get_logger_for_subgraph(model, first_node, last_node, cur_subgraph_idx, subgraph_candidate_idx, qconfig_str, OutputLogger, fqn)
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            assert not hasattr(model, attr_name)
            setattr(model, attr_name, logger_mod_orig)
            insertion_point = last_node
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(attr_name, args=(last_node,), kwargs={})
                insertion_point = logger
            cur_node_orig = first_node
            cur_node_copy = None
            first_node_copy = None
            while cur_node_orig in subgraph_to_use:
                if cur_node_orig is first_node:
                    new_args = cur_node_orig.args
                    new_kwargs = cur_node_orig.kwargs
                else:
                    first_arg_for_copy = cur_node_copy
                    new_args = tuple([first_arg_for_copy, *cur_node_orig.args[1:]])
                    new_kwargs = cur_node_orig.kwargs
                with model.graph.inserting_after(insertion_point):
                    cur_node_copy = model.graph.create_node(cur_node_orig.op, cur_node_orig.target, new_args, new_kwargs)
                    if first_node_copy is None:
                        first_node_copy = cur_node_copy
                if cur_node_orig != last_node:
                    assert len(cur_node_orig.users.keys()) == 1
                cur_node_orig = next(iter(cur_node_orig.users.keys()))
                assert not cur_node_orig.name.startswith(SHADOW_NODE_NAME_PREFIX)
                insertion_point = cur_node_copy
            subgraph_candidate_idx = 1
            logger_mod_orig = _get_logger_for_subgraph(model, first_node, last_node, cur_subgraph_idx, subgraph_candidate_idx, qconfig_str, OutputComparisonLogger, fqn)
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            assert not hasattr(model, attr_name)
            setattr(model, attr_name, logger_mod_orig)
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(attr_name, args=(cur_node_copy, last_node), kwargs={})
            orig_first_node_to_shadow_in_node[first_node] = first_node_copy
            orig_first_node_to_shadow_out_node[first_node] = cur_node_copy
        cur_subgraph_idx += 1
    model.recompile()
    nodes_to_skip = set()
    for n in orig_nodes:
        if n.op in ('placeholder', 'get_attr', 'output') or n in nodes_to_skip:
            continue
        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        if maybe_subgraph is not None:
            (first_node, last_node) = (maybe_subgraph[0], maybe_subgraph[-1])
            for node_to_skip in maybe_subgraph:
                nodes_to_skip.add(node_to_skip)
        else:
            (first_node, last_node) = (n, n)

        def maybe_remap_node_to_shadow(node):
            if False:
                while True:
                    i = 10
            '\n            If unshadowed `node` has a shadow version, return that. If not,\n            return `node`.\n            '
            if not isinstance(node, Node):
                return node
            if node.op in ('placeholder', 'get_attr'):
                return node
            prev_subgraph = _get_subgraph_containing_node(node, subgraphs_dedup)
            if prev_subgraph is None:
                prev_subgraph = [node]
            prev_first_node = prev_subgraph[0]
            prev_shadow_output = orig_first_node_to_shadow_out_node[prev_first_node]
            return prev_shadow_output
        cur_shadow_input = orig_first_node_to_shadow_in_node[first_node]
        assert cur_shadow_input is not None
        cur_shadow_input.args = tree_map(maybe_remap_node_to_shadow, cur_shadow_input.args)
        cur_shadow_input.kwargs = tree_map(maybe_remap_node_to_shadow, cur_shadow_input.kwargs)
        model.recompile()

def _get_weight_info_from_shadow_wrapper(shadow_wrapper: torch.nn.Module):
    if False:
        i = 10
        return i + 15
    placeholders_seen = 0
    for shadow_n in shadow_wrapper.graph.nodes:
        if shadow_n.op != 'placeholder':
            continue
        placeholders_seen += 1
        if placeholders_seen != 2:
            continue
        assert len(shadow_n.users) == 1
        quant_node = next(iter(shadow_n.users.keys()))
        new_args: Any = None
        if quant_node.target == torch.quantize_per_channel:
            (_weight, scale_node, zp_node, axis, dtype) = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, axis, dtype)
        else:
            assert quant_node.target == torch.quantize_per_tensor
            (_weight, scale_node, zp_node, dtype) = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, dtype)
        return (quant_node.target, new_args)
    return None

def extract_weight_comparison(m: GraphModule) -> NSResultsType:
    if False:
        i = 10
        return i + 15
    weighted_ops = {torch.nn.functional.linear}
    results: NSResultsType = {'model': {NSSingleResultValuesType.WEIGHT.value: {}}}
    for n in m.graph.nodes:
        if not (n.op == 'call_function' and n.target in weighted_ops):
            continue
        first_arg = n.args[0]
        shadow_wrapper_node = None
        for user in first_arg.users:
            if user.op == 'call_module' and user.target.startswith('shadow_wrapper'):
                shadow_wrapper_node = user
                break
        if shadow_wrapper_node is None:
            continue
        shadow_wrapper = getattr_from_fqn(m, shadow_wrapper_node.target)
        weight_info = _get_weight_info_from_shadow_wrapper(shadow_wrapper)
        if weight_info is None:
            continue
        w_node = n.args[1]
        w_obj = getattr_from_fqn(m, w_node.target).detach()
        (quant_fn, quant_fn_args_except_first) = weight_info
        new_args = (w_obj, *quant_fn_args_except_first)
        w_obj_q = quant_fn(*new_args)
        ref_node_name = n.name
        prev_node_name = n.name
        ref_node_type = get_target_type_str(n, m)
        prev_node_type = ref_node_type
        fqn = None
        if hasattr(m, '_node_name_to_scope'):
            fqn = m._node_name_to_scope[n.name][0]
        comparison = torch.ao.ns.fx.utils.compute_sqnr(w_obj, w_obj_q)
        result_fp32 = {'res_type': NSSingleResultValuesType.WEIGHT.value, 'values': [w_obj], 'prev_node_name': prev_node_name, 'prev_node_target_type': prev_node_type, 'ref_node_name': ref_node_name, 'ref_node_target_type': ref_node_type, 'index_within_arg': 0, 'index_of_arg': 0, 'fqn': fqn, 'qconfig_str': '', 'comparisons': [comparison], 'comparison_fn_name': 'sqnr'}
        result_q = {'res_type': NSSingleResultValuesType.WEIGHT.value, 'values': [w_obj_q], 'prev_node_name': prev_node_name, 'prev_node_target_type': prev_node_type, 'ref_node_name': ref_node_name, 'ref_node_target_type': ref_node_type, 'index_within_arg': 0, 'index_of_arg': 0, 'fqn': fqn, 'qconfig_str': '', 'comparisons': [comparison], 'comparison_fn_name': 'sqnr'}
        (_1, _2, node_idx, _3) = shadow_wrapper_node.target.split('_')
        name_fp32 = f'subgraph_{node_idx}_0'
        name_q = f'subgraph_{node_idx}_1'
        results['model'][NSSingleResultValuesType.WEIGHT.value][name_fp32] = [result_fp32]
        results['model'][NSSingleResultValuesType.WEIGHT.value][name_q] = [result_q]
    return results

def group_results_by_subgraph(results: NSResultsType) -> Any:
    if False:
        i = 10
        return i + 15
    "\n    Creates a comparison of results\n\n    Input:\n\n    {\n      'model': {\n        'node_output': {\n          'subgraph_0_0': [\n            'values': [torch.tensor(...), ...], ...\n            'ref_node_name': ...,\n            'ref_node_target_type': ...,\n            'qconfig_str': ...,\n            'comparisons': [], ...\n            'comparison_fn_name': '',\n            'fqn': '...',\n          ],\n          'subgraph_0_1': [\n            'values': [torch.tensor(...), ...], ...\n            'ref_node_name': ...,\n            'ref_node_target_type': ...,\n            'qconfig_str': ...,\n            'comparisons': [torch.tensor(...), ...], ...\n            'comparison_fn_name': '...',\n            'fqn': '...',\n          ],\n          ...\n        },\n      },\n    }\n\n    Output:\n    {\n      'subgraph_0': {\n        '0': {\n          'ref_node_name': '...',\n          'ref_node_target_type': ...,\n          'values': [torch.tensor(...), ...],\n          'qconfig_str': None,\n          'comparisons': [torch.tensor(...), ...], ...\n          'comparison_fn_name': '...',\n          'fqn': '...',\n        },\n        '1': {\n          'ref_node_name': '...',\n          'ref_node_target_type': ...,\n          'values': [torch.tensor(...), ...],\n          'qconfig_str': '...',\n          'comparisons': [torch.tensor(...), ...], ...\n          'comparison_fn_name': '...',\n          'fqn': '...',\n        },\n      },\n    }\n\n    "
    subgraph_name_to_subgraph_results: Any = collections.defaultdict(dict)
    key_to_use = next(iter(results['model'].keys()))
    for (subgraph_name_with_idx, subgraph_candidate_results) in results['model'][key_to_use].items():
        (subgraph_str, subgraph_idx, subgraph_candidate_idx) = subgraph_name_with_idx.split('_')
        subgraph_name = f'{subgraph_str}_{subgraph_idx}'
        subgraph_results = {'ref_node_name': subgraph_candidate_results[0]['ref_node_name'], 'ref_node_target_type': subgraph_candidate_results[0]['ref_node_target_type'], 'fqn': subgraph_candidate_results[0]['fqn'], 'values': subgraph_candidate_results[0]['values'], 'qconfig_str': subgraph_candidate_results[0]['qconfig_str'], 'comparisons': subgraph_candidate_results[0]['comparisons'], 'comparison_fn_name': subgraph_candidate_results[0]['comparison_fn_name']}
        subgraph_name_to_subgraph_results[subgraph_name][subgraph_candidate_idx] = subgraph_results
    return dict(subgraph_name_to_subgraph_results)

def create_results_comparison(results_grouped) -> Any:
    if False:
        print('Hello World!')
    "\n    Input:\n\n    {\n      'subgraph_0': {\n        '0': {\n          'ref_node_name': '...',\n          'ref_node_target_type': ...,\n          'values': [torch.tensor(...), ...],\n          'qconfig_str': '',\n          'comparisons': [],\n          'comparison_fn_name': '',\n          'fqn': '...',\n        },\n        '1': {\n          'ref_node_name': '...',\n          'ref_node_target_type': ...,\n          'values': [torch.tensor(...), ...],\n          'qconfig_str': '...',\n          'comparisons': [torch.tensor(...), ...],\n          'comparison_fn_name': 'sqnr',\n          'fqn': '...',\n        },\n      },\n    }\n\n    Output:\n    {\n      'subgraph_0': {\n        'ref_node_name': '...',\n        'ref_node_target_type': '...',\n        'fqn': '...',\n        'candidates': {\n          '1': {\n            'qconfig_str': ...,\n            'comparison_fn_name': 'sqnr',\n            'cmp_raw': [..., ...],\n            'cmp_mean': ...,\n          },\n          ...,\n        },\n      },\n    }\n    "
    results_comparison = {}
    for (subgraph_name, subgraph_results) in results_grouped.items():
        candidates = {}
        for (subgraph_inner_name, subgraph_inner_result) in subgraph_results.items():
            if subgraph_inner_name == '0':
                continue
            cmp_raw = subgraph_inner_result['comparisons']
            cmp_raw_tensor = torch.stack(cmp_raw)
            candidates[subgraph_inner_name] = {'qconfig_str': subgraph_inner_result['qconfig_str'], 'comparison_fn_name': subgraph_inner_result['comparison_fn_name'], 'cmp_raw': cmp_raw_tensor, 'cmp_mean': torch.mean(cmp_raw_tensor)}
        results_comparison[subgraph_name] = {'ref_node_name': subgraph_results['0']['ref_node_name'], 'ref_node_target_type': subgraph_results['0']['ref_node_target_type'], 'fqn': subgraph_results['0']['fqn'], 'candidates': candidates}
    return results_comparison

def print_n_shadows_summary(results_comparison) -> None:
    if False:
        return 10
    "\n    Input:\n\n    {\n      'subgraph_0': {\n        'ref_node_name': 'linear1',\n        'ref_node_target_type': '...',\n        'fqn': '...',\n        'candidates': {\n          '1': {\n            'qconfig_str': ...,\n            'comparison_fn_name': ...,\n            'cmp_raw': [45.0, 55.0],\n            'cmp_mean': 50.0,\n          },\n          ...,\n        },\n      },\n    }\n\n    Prints:\n\n    node_name | node_type | fqn | 0    | 1    | ...\n    linear1   | ...       | ... | 45.0 | 50.0 | ...\n    "
    try:
        from tabulate import tabulate
    except ImportError:
        print('`print_tabular` relies on the library `tabulate`, which could not be found on this machine. Run `pip install tabulate` to install the library.')
        return
    results = []
    for subgraph_data in results_comparison.values():
        mean_all_candidates = [candidate['cmp_mean'] for (candidate_name, candidate) in subgraph_data['candidates'].items()]
        data_row = [subgraph_data['ref_node_name'], subgraph_data['ref_node_target_type'], subgraph_data['fqn'], *mean_all_candidates]
        results.append(data_row)
    max_candidate_idx_len = -1
    for data_row in results:
        max_candidate_idx_len = max(max_candidate_idx_len, len(data_row[1]))
    candidate_idx_headers = [str(x) for x in range(max_candidate_idx_len)]
    headers = ['node_name', 'node_type', 'fqn', *candidate_idx_headers]
    print(tabulate(results, headers=headers))