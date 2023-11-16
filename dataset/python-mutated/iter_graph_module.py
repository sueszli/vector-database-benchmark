import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import clone_subgraph, get_output, is_leaf_subgraph
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
logger: logging.Logger = logging.getLogger('IterGraphModule')

class IterGraph(fx.Graph):
    """``IterGraph`` is used to perform cross-iteration optimization.

    ``IterGraph`` keeps track of the 3 graphs, self (the original graph), setup graph, and
    cleanup graph. The 3 graphs should be identical copies of a ``fx.Graph``.

    IterGraph subclass fx.Graph to override the necessary APIs that will be used
    when constructing a optimization, e.g., communication fusion. IterGraph also
    provides APIs that originally belong to fx.Node and all these APIs will have
    ``node_`` prefix. For example, ``IterGraph.node_prepend`` is the equivalence
    of ``fx.Node.prepend``. Note that all the optimizations must be constructed
    using these APIs.
    """

    def __init__(self, orig_graph: fx.Graph, setup_graph: fx.Graph, cleanup_graph: fx.Graph, owning_module: Optional[fx.GraphModule]=None, tracer_cls: Optional[Type['fx.Tracer']]=None, tracer_extras: Optional[Dict[str, Any]]=None):
        if False:
            while True:
                i = 10
        super().__init__(owning_module, tracer_cls, tracer_extras)
        output_vals = self.graph_copy(orig_graph, {}, return_output_node=True)
        self._codegen = copy.deepcopy(orig_graph._codegen)
        assert isinstance(output_vals, tuple)
        (output_val, old_output_val) = output_vals
        super().output(output_val, type_expr=getattr(old_output_val, 'type', None))
        self.setup_graph = setup_graph
        self.cleanup_graph = cleanup_graph
        self._all_graphs: Tuple[fx.Graph, ...] = (self.setup_graph, self.cleanup_graph, cast(fx.Graph, super()))
        self._setup_mapping: Dict[fx.Node, fx.Node] = {}
        self._cleanup_mapping: Dict[fx.Node, fx.Node] = {}
        self._freeze_cross_iter_movement = False
        self._cross_iter_block_count = 0
        for (node, setup_node, cleanup_node) in zip(self.nodes, self.setup_graph.nodes, self.cleanup_graph.nodes):
            self._setup_mapping[node] = setup_node
            self._cleanup_mapping[node] = cleanup_node
        self.num_extra_output = 0

    def _lookup_node(self, node: fx.Node, graph: fx.Graph) -> Optional[fx.Node]:
        if False:
            for i in range(10):
                print('nop')
        if graph == self.setup_graph:
            return self._setup_mapping.get(node, None)
        elif graph == self.cleanup_graph:
            return self._cleanup_mapping.get(node, None)
        return node

    def _fx_graph_call(self, graph: fx.Graph, func: str, *args: Any, **kwargs: Any) -> Any:
        if False:
            print('Hello World!')
        fx_graph: fx.Graph = graph if graph != self else cast(fx.Graph, super())
        return getattr(fx_graph, func)(*args, **kwargs)

    def _insert_context(self, func: str, node: fx.Node):
        if False:
            while True:
                i = 10

        class _InsertPoint:

            def __init__(self, insert_points: List[Any]):
                if False:
                    while True:
                        i = 10
                self.insert_points = insert_points

            def __enter__(self):
                if False:
                    return 10
                pass

            def __exit__(self, type, value, tb):
                if False:
                    print('Hello World!')
                for insert_point in self.insert_points:
                    insert_point.__exit__(type, value, tb)
        insert_points = []
        for graph in self._all_graphs:
            if node:
                actual_node = self._lookup_node(node, graph)
                assert actual_node is not None, 'Cannot handle None case now.'
            else:
                actual_node = node
            insert_points.append(getattr(graph, func)(actual_node))
        return _InsertPoint(insert_points)

    def inserting_after(self, node):
        if False:
            for i in range(10):
                print('nop')
        if self._freeze_cross_iter_movement:
            return super().inserting_after(node)
        return self._insert_context('inserting_after', node)

    def inserting_before(self, node):
        if False:
            print('Hello World!')
        if self._freeze_cross_iter_movement:
            return super().inserting_before(node)
        return self._insert_context('inserting_before', node)

    def _forward_subgraph_inputs(self, subgraph: List[fx.Node], graph: fx.Graph, erase_node: bool) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Turn the inputs of a subgraph into the extra output of the entire graph.\n\n        If ``erase_node`` is True, the subgraph will be erased from the graph -- essentially forward the inputs\n        of the subgraph to the output of the graph.\n        '
        output = get_output(graph)
        inputs = []
        all_nodes: Set[fx.Node] = set(subgraph)
        for node in subgraph:
            node_inputs = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            for _input in node_inputs:
                if not isinstance(_input, fx.Node):
                    continue
                if _input in all_nodes:
                    continue
                inputs.append(_input)
        if erase_node:
            erased = set()
            for node in reversed(subgraph):
                if len(node.users) == 1:
                    key = next(iter(node.users.keys()))
                    if key == output:
                        (flatten_args, spec) = tree_flatten((output.args, output.kwargs))
                        if node not in flatten_args:
                            node.users.clear()
                        elif str(node.target).startswith('aten.copy_'):
                            for i in range(len(flatten_args)):
                                if flatten_args[i] == node:
                                    flatten_args[i] = node.args[0]
                        else:
                            raise RuntimeError(f'IterGraph does not how to forward the output of {node}')
                        (output.args, output.kwargs) = tree_unflatten(flatten_args, spec)
                for user in list(node.users.keys()):
                    if user in erased:
                        node.users.pop(user)
                if node.users:
                    raise RuntimeError(f'IterGraph has not supported moving the nodes that produce users output result. Error node: {node}.')
                self._fx_graph_call(graph, 'erase_node', node)
                erased.add(node)
        if self.num_extra_output:
            cast(List[fx.Node], output.args[0][-1]).extend(inputs)
            new_output = output.args[0]
        elif isinstance(graph._codegen, _PyTreeCodeGen):
            codegen = graph._codegen
            new_output = list(output.args[0])
            new_output.append(inputs)
            assert codegen.pytree_info.out_spec is not None
            original_tree_out = tree_unflatten(cast(List[Any], output.args[0]), codegen.pytree_info.out_spec)
            (_, out_spec) = tree_flatten((original_tree_out, None))
            codegen.pytree_info = codegen.pytree_info._replace(out_spec=out_spec)
        else:
            new_output = (output.args[0], inputs)
        self._fx_graph_call(graph, 'erase_node', output)
        self._fx_graph_call(graph, 'output', new_output)
        logger.info('Extended outputs from the subgraph inputs: %s', str(inputs))
        return len(inputs)

    def _forward_inputs_to_subgraph(self, subgraph: List[fx.Node], graph: fx.Graph, extra_input: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create extra input nodes and forward the input nodes to the ``subgraph``.\n\n        The external input nodes of ``subgraph`` (nodes that are not in ``subgraph``) will replaced by the newly\n        created input nodes.\n        '
        placeholders = [node for node in graph.nodes if str(node.op) == 'placeholder']
        assert placeholders, 'No placeholders are found'
        with self._fx_graph_call(graph, 'inserting_after', placeholders[-1]):
            new_input_nodes = list(reversed([self._fx_graph_call(graph, 'placeholder', f'cross_iter_input_{self._cross_iter_block_count}_{i}') for i in reversed(range(extra_input))]))
        all_nodes = set(subgraph)
        new_input_index = 0
        for node in subgraph:
            (node_inputs, spec) = tree_flatten((node.args, node.kwargs))
            new_node_inputs = []
            for input_node in node_inputs:
                if not isinstance(input_node, fx.Node) or input_node in all_nodes:
                    new_node_inputs.append(input_node)
                else:
                    new_node_inputs.append(new_input_nodes[new_input_index])
                    new_input_index += 1
            (node.args, node.kwargs) = tree_unflatten(new_node_inputs, spec)
        assert new_input_index == len(new_input_nodes), f'More inputs than needed {len(new_input_nodes)} > {new_input_index}'
        if isinstance(graph._codegen, _PyTreeCodeGen) and graph._codegen.pytree_info.in_spec is not None:
            codegen = graph._codegen
            original_tree_in = tree_unflatten(placeholders, codegen.pytree_info.in_spec)
            (_, in_spec) = tree_flatten(tuple(list(original_tree_in) + new_input_nodes))
            codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)
            for new_input in new_input_nodes:
                codegen.pytree_info.orig_args.append(new_input.name)
            codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)

    def move_to_next_iter_before(self, subgraph: List[fx.Node], target_node: fx.Node) -> None:
        if False:
            i = 10
            return i + 15
        'Move the ``subgraph`` to the next iteration before ``target_node``.\n\n        The ``subgraph`` is a list of fx.Node and must satisfy the following\n        restrictions:\n            1. The order of the nodes in ``subgraph`` must obey the topological\n               sort order.\n            2. The users of the node in ``subgraph`` must be one of the following:\n                a.) the user is also a node in ``subgraph``.\n                b.) the user is the output of the full graph.\n                c.) the node has users (side effect node).\n        '
        if self._freeze_cross_iter_movement:
            raise RuntimeError('The cross-iteration movement has been frozen for the given IterGraph.')
        if not is_leaf_subgraph(self, subgraph):
            raise ValueError('The target nodes for ``move_to_next_iter_before`` must satisfy one of the following conditions: 1) the user of the node is in the target nodes, 2) the user is the ouput of the graph, 3) there are no users -- the node is a side-effect node. ')
        self._cross_iter_block_count += 1
        setup_subgraph: List[fx.Node] = []
        for node in subgraph:
            mapped_node = self._lookup_node(node, self.setup_graph)
            assert mapped_node is not None
            setup_subgraph.append(mapped_node)
        setup_extra_input = self._forward_subgraph_inputs(subgraph=setup_subgraph, graph=self.setup_graph, erase_node=True)
        target_cleanup_node = self._lookup_node(target_node, self.cleanup_graph)
        assert target_cleanup_node is not None, 'The target_cleanup_node is None.'
        cleanup_subgraph: List[fx.Node] = []
        for node in subgraph:
            mapped_node = self._lookup_node(node, self.cleanup_graph)
            assert mapped_node is not None
            cleanup_subgraph.append(mapped_node)
        cloned_subgraph = clone_subgraph(self.cleanup_graph, cleanup_subgraph, target=target_cleanup_node)
        self._forward_inputs_to_subgraph(cloned_subgraph, self.cleanup_graph, setup_extra_input)
        main_extra_input = self._forward_subgraph_inputs(subgraph=subgraph, graph=self, erase_node=False)
        assert main_extra_input == setup_extra_input
        for node in subgraph:
            target_node.prepend(node)
        self._forward_inputs_to_subgraph(subgraph, self, main_extra_input)
        for node in self.cleanup_graph.nodes:
            if len(node.users) == 0:
                node.users['__hold__'] = None
        for node in self.nodes:
            if len(node.users) == 0:
                node.users['__hold__'] = None
        self.num_extra_output += main_extra_input

    def move_before(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        if False:
            i = 10
            return i + 15
        for graph in self._all_graphs:
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None
            for actual_node in actual_nodes:
                actual_target_node.prepend(actual_node)

    def move_after(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        if False:
            i = 10
            return i + 15
        for graph in self._all_graphs:
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            actual_target_node = self._lookup_node(target_node, graph)
            for actual_node in actual_nodes:
                assert actual_target_node is not None
                actual_target_node.append(actual_node)
                actual_target_node = actual_node

    def call_function(self, the_function: Callable[..., Any], args: Optional[Tuple[Argument, ...]]=None, kwargs: Optional[Dict[str, Argument]]=None, type_expr: Optional[Any]=None) -> fx.Node:
        if False:
            return 10
        if self._freeze_cross_iter_movement:
            return super().call_function(the_function, args, kwargs, type_expr)
        setup_args = tree_map(lambda arg: self._lookup_node(arg, self.setup_graph) if isinstance(arg, fx.Node) else arg, args)
        setup_kwargs = tree_map(lambda arg: self._lookup_node(arg, self.setup_graph) if isinstance(arg, fx.Node) else arg, kwargs)
        cleanup_args = tree_map(lambda arg: self._lookup_node(arg, self.cleanup_graph) if isinstance(arg, fx.Node) else arg, args)
        cleanup_kwargs = tree_map(lambda arg: self._lookup_node(arg, self.cleanup_graph) if isinstance(arg, fx.Node) else arg, kwargs)
        setup_node = self.setup_graph.call_function(the_function, setup_args, setup_kwargs, type_expr)
        main_node = super().call_function(the_function, args, kwargs, type_expr)
        cleanup_node = self.cleanup_graph.call_function(the_function, cleanup_args, cleanup_kwargs, type_expr)
        self._setup_mapping[main_node] = setup_node
        self._cleanup_mapping[main_node] = cleanup_node
        return main_node

    def erase_node(self, to_erase: fx.Node) -> None:
        if False:
            i = 10
            return i + 15
        if self._freeze_cross_iter_movement:
            return super().erase_node(to_erase)
        setup_node = self._lookup_node(to_erase, self.setup_graph)
        assert setup_node is not None, 'setup_node is None'
        self.setup_graph.erase_node(setup_node)
        super().erase_node(to_erase)
        cleanup_node = self._lookup_node(to_erase, self.cleanup_graph)
        self.cleanup_graph.erase_node(cleanup_node)

    def placeholder(self, name: str, type_expr: Optional[Any]=None, default_value: Any=inspect.Signature.empty) -> fx.Node:
        if False:
            print('Hello World!')
        if self._freeze_cross_iter_movement:
            return super().placeholder(name, type_expr, default_value)
        main_placeholder = super().placeholder(name, type_expr, default_value)
        setup_placeholder = self.setup_graph.placeholder(name, type_expr, default_value)
        cleanup_placeholder = self.cleanup_graph.placeholder(name, type_expr, default_value)
        self._setup_mapping[main_placeholder] = setup_placeholder
        self._cleanup_mapping[main_placeholder] = cleanup_placeholder
        return main_placeholder

    def output(self, result: Argument, type_expr: Optional[Any]=None) -> fx.Node:
        if False:
            while True:
                i = 10
        if self._freeze_cross_iter_movement:
            return super().output(result, type_expr)
        main_output = super().output(result, type_expr)
        setup_result = tree_map(lambda _result: self._lookup_node(_result, self.setup_graph) if isinstance(_result, fx.Node) else _result, result)
        cleanup_result = tree_map(lambda _result: self._lookup_node(_result, self.cleanup_graph) if isinstance(_result, fx.Node) else _result, result)
        self.setup_graph.output(setup_result, type_expr)
        self.cleanup_graph.output(cleanup_result, type_expr)
        return main_output

    def lint(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.setup_graph.lint()
        super().lint()
        self.cleanup_graph.lint()

    def node_prepend(self, target_node: fx.Node, node: fx.Node) -> None:
        if False:
            i = 10
            return i + 15
        'Prepend node to target_node.'
        if self._freeze_cross_iter_movement:
            target_node.prepend(node)
            return
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, 'The node is None'
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None, 'The target node is None'
            actual_target_node.prepend(actual_node)

    def node_append(self, target_node: fx.Node, node: fx.Node) -> None:
        if False:
            while True:
                i = 10
        'Append node to target_node.'
        if self._freeze_cross_iter_movement:
            target_node.append(node)
            return
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, f'The actual node is None, {node}.'
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None, f'The actual target node is None, {target_node}.'
            actual_target_node.append(actual_node)

    def node_set_args(self, node: fx.Node, args: Tuple[Argument, ...]) -> None:
        if False:
            i = 10
            return i + 15
        if self._freeze_cross_iter_movement:
            node.args = args
            return
        setup_args = tree_map_only(fx.Node, lambda _arg: self._lookup_node(_arg, self.setup_graph), args)
        setup_node = self._lookup_node(node, self.setup_graph)
        assert setup_node is not None
        setup_node.args = setup_args
        cleanup_args = tree_map_only(fx.Node, lambda _arg: self._lookup_node(_arg, self.cleanup_graph), args)
        cleanup_node = self._lookup_node(node, self.cleanup_graph)
        assert cleanup_node is not None
        cleanup_node.args = cleanup_args
        node.args = args

    def node_set_kwargs(self, node: fx.Node, kwargs: Dict[str, Argument]) -> None:
        if False:
            print('Hello World!')
        if self._freeze_cross_iter_movement:
            node.kwargs = kwargs
            return
        setup_kwargs = tree_map_only(fx.Node, lambda _arg: self._lookup_node(_arg, self.setup_graph), kwargs)
        setup_node = self._lookup_node(node, self.setup_graph)
        assert setup_node is not None
        setup_node.kwargs = setup_kwargs
        cleanup_kwargs = tree_map_only(fx.Node, lambda _arg: self._lookup_node(_arg, self.cleanup_graph), kwargs)
        cleanup_node = self._lookup_node(node, self.cleanup_graph)
        assert cleanup_node is not None
        cleanup_node.kwargs = cleanup_kwargs
        node.kwargs = kwargs

    def node_replace_all_uses_with(self, node: fx.Node, replace_with: fx.Node, delete_user_cb: Callable[[fx.Node], bool]=lambda user: True, *, propagate_meta=False) -> List[fx.Node]:
        if False:
            i = 10
            return i + 15
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            actual_replace_with = self._lookup_node(replace_with, graph)
            assert actual_node is not None
            ret = actual_node.replace_all_uses_with(actual_replace_with, delete_user_cb, propagate_meta=propagate_meta)
        return ret

    def node_add_user(self, node: fx.Node, user: Any) -> None:
        if False:
            i = 10
            return i + 15
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            assert actual_node is not None
            actual_node.users[actual_user_node] = None

    def node_remove_user(self, node: fx.Node, user: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            assert actual_node is not None
            del actual_node.users[actual_user_node]

    def keep_unused_nodes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for node in self.nodes:
            if len(node.users) == 0 and str(node.op) != 'output':
                self.node_add_user(node, '__hold__')

    def functionalize_optim(self) -> None:
        if False:
            return 10
        for node in reversed(self.nodes):
            if node.name.startswith('output'):
                output_node = node
            elif node.name.startswith('_fused_adam_'):
                optim_node = node
            elif node.name.startswith('_foreach_add_'):
                step_node = node
                self.node_add_user(optim_node, output_node)
                self.node_add_user(step_node, optim_node)

    def defunctionalize_optim(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for graph in self._all_graphs:
            for node in reversed(graph.nodes):
                if node.name.startswith('output'):
                    output_node = node
                elif node.name.startswith('_fused_adam_'):
                    optim_node = node
                elif node.name.startswith('_foreach_add_'):
                    step_node = node
                    optim_node.users.pop(output_node, None)
                    step_node.users.pop(optim_node, None)

    def freeze_cross_iter_movement(self) -> None:
        if False:
            while True:
                i = 10
        self._freeze_cross_iter_movement = True

class IterGraphModule(nn.Module):
    """``IterGraphModule`` provides the ability to do cross-iteration optimization.

    Given a ``fx.GraphModule``, main_gm, ``IterGraphModule`` internally
    duplicate it to 3 copies and redirect the ``forward`` request to a different
    ``fx.GraphModule`` based on the iteration count. This allows users to do
    graph optimizations that across iterations (e.g., moving collective wait in
    the backward to the forward of the next iteration).

    Note that users must call the APIs provided by ``IterGraphModule`` or
    ``IterGraph`` to rewrite the graph so that ``IterGraphModule`` can keep the
    data dependency for all 3 graphs.
    """

    def __init__(self, main_gm: fx.GraphModule, max_iters: int=-1, enable_inductor: bool=False) -> None:
        if False:
            print('Hello World!')
        super().__init__()

        def _copy_gm(src: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
            if False:
                return 10
            gm = fx.GraphModule(src, graph)
            gm.meta = getattr(graph, 'meta', {})
            return gm
        self.setup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.cleanup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.main_gm = _copy_gm(main_gm, IterGraph(main_gm.graph, self.setup_gm.graph, self.cleanup_gm.graph))
        self._iter = 0
        self._max_iters = max_iters
        self._previous_output: Tuple[Any, ...] = tuple()
        self._num_extra_output = 0
        self._is_frozen = False
        self._enable_inductor = enable_inductor

    def finalize_setup(self) -> None:
        if False:
            while True:
                i = 10
        'Set up the internal states and also get the signal from users that what is the maximum iteration count.\n\n        This method must be called before the forward() is called.\n        '
        if not self._is_frozen:
            self.graph.freeze_cross_iter_movement()
            self._num_extra_output = self.graph.num_extra_output
            if self._enable_inductor:
                self.main_gm = partial_lower(self.main_gm)
            self._is_frozen = True
        self._iter = 0

    def _run(self, gm: fx.GraphModule, last_iter: bool, *args, **kwargs) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if self._num_extra_output > 0:
            new_args = args + self._previous_output
            output = gm(*new_args, **kwargs)
            if not last_iter:
                assert len(output) == 2
                self._previous_output = tuple(output[-1])
                assert len(self._previous_output) > 0, 'There should be at least one extra output.'
                output = output[0]
        else:
            output = gm(*args, **kwargs)
        return output

    def forward(self, *args: Any, last_iter: bool=False, **kwargs: Any) -> Any:
        if False:
            return 10
        self._iter += 1
        last_iter = last_iter or self._iter == self._max_iters
        if last_iter:
            logger.info('Using the cleanup graph')
            gm = self.cleanup_gm
            profiler_string = '## IterGraphModule: Cleanup Graph ##'
            self._iter = 0
        elif self._iter == 1:
            logger.info('Using the setup graph')
            gm = self.setup_gm
            profiler_string = '## IterGraphModule: Setup Graph ##'
        else:
            gm = self.main_gm
            if self._iter == 2:
                logger.info('Using the main graph')
                profiler_string = '## IterGraphModule -- Maybe Compiling ##'
            else:
                profiler_string = '## IterGraphModule ##'
        with record_function(profiler_string):
            return self._run(gm, last_iter, *args, **kwargs)

    @property
    def graph(self) -> IterGraph:
        if False:
            print('Hello World!')
        return cast(IterGraph, self.main_gm.graph)

    def recompile(self) -> PythonCode:
        if False:
            i = 10
            return i + 15
        self.setup_gm.recompile()
        self.cleanup_gm.recompile()
        return self.main_gm.recompile()

    def freeze_cross_iter_movement(self) -> None:
        if False:
            print('Hello World!')
        self.graph.freeze_cross_iter_movement()
        self._num_extra_output = self.graph.num_extra_output

    def print_readable(self, print_output: bool=True) -> str:
        if False:
            return 10
        return self.main_gm.print_readable(print_output)

    def print_all_graphs(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.info('Printing the three fx.Graph:')
        logger.info('1. Setup fx.Graph:')
        logger.info('%s', self.setup_gm.graph)
        logger.info('2. Main fx.Graph:')
        logger.info('%s', self.main_gm.graph)
        logger.info('3. Cleanup fx.Graph:')
        logger.info('%s', self.cleanup_gm.graph)

    def print_all_graph_modules(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.info('Printing the three fx gm:')
        logger.info('1. Setup fx.GraphModule:')
        logger.info('%s', self.setup_gm.print_readable(False))
        logger.info('2. Main fx.GraphModule:')
        logger.info('%s', self.main_gm.print_readable(False))
        logger.info('3. Cleanup fx.GraphModule:')
        logger.info('%s', self.cleanup_gm.print_readable(False))