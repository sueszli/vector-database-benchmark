import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, cast, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Union
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import CommType, dump_graphs_to_files, find_node, get_output, OP
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
logger: logging.Logger = logging.getLogger('graph_optimization')
aten = torch.ops.aten
fake_tensor_mode = FakeTensorMode()
_optimized_func: Set[str] = set()
_prerequisite_sets: DefaultDict[str, Set[str]] = collections.defaultdict(set)
_apply_before_sets: DefaultDict[str, Set[str]] = collections.defaultdict(set)
_dump_graph_folder: str = ''

def enable_graph_optimization_dump(folder: str=''):
    if False:
        print('Hello World!')
    global _dump_graph_folder
    if not folder:
        folder = tempfile.mkdtemp()
    _dump_graph_folder = folder

def graph_optimization_pass(prerequisites: Iterable[Callable], apply_after: Iterable[Callable]) -> Callable:
    if False:
        return 10
    'Define the contract of a graph optimization pass.\n\n    All the passes should be wrapped with this decorator.\n    `prerequisites` is used to annotate the prerequisite passes of the this pass.\n    `apply_after` means that this wrapped pass must be applied after the passes\n    in `apply_after`. The difference between `prerequisites` and `apply_after`\n    is that all the passes in `prerequisites` must be applied to the graph and\n    must be applifed before the wrapped pass while the passes `apply_after` are\n    optional. But if a pass in `apply_after` is applied to the graph, it has to\n    be done before the wrapped pass.\n    Optimizer pass developers are required to add these fields accordingly and\n    users need to follow the restrictions to avoid the assert.\n\n    Current design has one limitation: users can only apply the optimizations\n    once.  In some cases, we may need to run multiple the same optimization\n    multiple time, e.g., optimization passes -> profiling the result -> apply\n    optimization passes with the profiling result again. This limitation will be\n    addressed limitation in the future.\n\n    Args:\n        prerequisites (Iterable[Callable]): the list of string to the names of\n            passes which are the prerequisites of this pass.\n        apply_after (Iterable[Callable]): the list of string to the names of\n            passes that can not be applied after the wrapped pass.\n    '

    def inner(func: Callable) -> Callable:
        if False:
            print('Hello World!')

        def make_key(func: Callable) -> str:
            if False:
                return 10
            return f'{func.__module__}.{func.__name__}'
        func_key = make_key(func)
        _prerequisite_sets[func_key] = {make_key(f) for f in prerequisites}
        for apply_after_pass in apply_after:
            _apply_before_sets[make_key(apply_after_pass)].add(func_key)

        @wraps(func)
        def pass_wrapper(gm: Union[fx.GraphModule, IterGraphModule], *args: Any, **kwargs: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            begin = time.time()
            assert isinstance(gm, (fx.GraphModule, IterGraphModule)), 'The first argument of the pass must be either fx.GraphModule or IterGraphModule.'
            assert func_key not in _optimized_func, f'Cannot apply {func_key} twice.'
            invalid_passes = _apply_before_sets[func_key].intersection(_optimized_func)
            assert not invalid_passes, f'{invalid_passes} must be applied after {func_key}.'
            assert _prerequisite_sets[func_key].issubset(_optimized_func), f'{_prerequisite_sets[func_key] - _optimized_func} are the prerequisites of {func_key} but are not applified. Applied passes are {_optimized_func}.'
            func(gm, *args, **kwargs)
            gm.graph.lint()
            gm.graph.eliminate_dead_code()
            gm.recompile()
            _optimized_func.add(func_key)
            prefix = f'after_{func.__name__}'
            if _dump_graph_folder:
                if isinstance(gm, IterGraphModule):
                    dump_graphs_to_files({f'{prefix}_setup_gm': gm.setup_gm, f'{prefix}_main_gm': gm.main_gm, f'{prefix}_cleanup_gm': gm.cleanup_gm}, _dump_graph_folder)
                else:
                    dump_graphs_to_files({prefix: gm}, _dump_graph_folder)
            logger.info('Spent %f seconds applying %s', time.time() - begin, func_key)
        return pass_wrapper
    return inner

@dataclass(unsafe_hash=True)
class CommBlock:
    shape: Optional[torch.Size]
    node_list: List[fx.Node]
    inputs: List[fx.Node]
    wait_nodes: List[fx.Node]
    comm_node: fx.Node
    outputs: Set[fx.Node]

def get_comm_block(comm_node: fx.Node) -> CommBlock:
    if False:
        for i in range(10):
            print('nop')
    'Find out all the nodes belong to this communcation given a collective node (e.g., allreduce).\n\n    Args:\n        comm_node(fx.Node): The target communication/collective node.\n\n    Returns:\n        The CommBlock that encapsulates the related nodes (e.g., wait_node) of\n        the given comm_node.\n    '
    MAX_WAIT_DISTANCE = 5
    node_list = []
    wait_nodes = []
    inputs = pytree.arg_tree_leaves(*comm_node.args, **comm_node.kwargs)
    input_nodes = [inp for inp in inputs if isinstance(inp, fx.Node)]
    distance = 0
    wait_prefixes = ('wait_comm', 'wait_tensor')
    non_end_users_nodes = ('split', 'reshape', 'getitem', 'detach', 'alias')
    nodes = collections.deque([comm_node, None])
    while nodes and distance < 5:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        node_list.append(node)
        if node.name.startswith(wait_prefixes):
            wait_nodes.append(node)
        else:
            for child in node.users:
                if isinstance(child, fx.Node):
                    nodes.append(child)
    if not wait_nodes:
        raise RuntimeError('The wait nodes are too far away from the comm node {comm_node}.')
    outputs: Set[fx.Node] = set()
    nodes = collections.deque(wait_nodes)
    while nodes:
        node = nodes.popleft()
        assert node is not None
        for user in node.users:
            if isinstance(user, fx.Node) and user.name.startswith(non_end_users_nodes):
                nodes.append(user)
                node_list.append(user)
            else:
                outputs.add(node)
                break
    tensor_meta = input_nodes[0].meta.get('tensor_meta', None)
    return CommBlock(shape=torch.Size((int(s) for s in tensor_meta.shape)) if tensor_meta else None, node_list=node_list, wait_nodes=wait_nodes, comm_node=comm_node, inputs=input_nodes, outputs=outputs)

def get_all_comm_blocks(gm: IterGraphModule, comm_ops: Union[Tuple[str, ...], str]) -> List[CommBlock]:
    if False:
        while True:
            i = 10
    return [get_comm_block(node) for node in gm.graph.nodes if node.name.startswith(comm_ops)]

def _create_meta_val(fake_tensor_mode: FakeTensorMode, val: FakeTensor) -> FakeTensor:
    if False:
        while True:
            i = 10
    return FakeTensor(fake_tensor_mode, torch.empty(val.shape, dtype=val.dtype, device='meta', requires_grad=val.requires_grad), val.device)

def _create_meta_tensor_meta(fake_tensor_mode: FakeTensorMode, val: FakeTensor) -> TensorMetadata:
    if False:
        while True:
            i = 10
    return TensorMetadata(shape=val.shape, dtype=val.dtype, requires_grad=val.requires_grad, stride=val.stride, memory_format=None, is_quantized=False, qparams={})

def _call_function(gm: IterGraphModule, fake_tensor_mode: FakeTensorMode, meta_val: Optional[FakeTensor], function: Any, *args: Any, **kwargs: Any) -> fx.Node:
    if False:
        return 10
    node = gm.graph.call_function(function, args, kwargs)
    if meta_val is None:
        (flat_args, spec) = tree_flatten((args, kwargs))
        new_flat_args = []
        memory_format = None
        for arg in flat_args:
            if not isinstance(arg, fx.Node):
                new_flat_args.append(arg)
                continue
            val = arg.meta['val']
            new_flat_args.append(_create_meta_val(fake_tensor_mode, val))
        (fake_args, fake_kwargs) = tree_unflatten(new_flat_args, spec)
        new_meta_val = function(*fake_args, **fake_kwargs)
    else:
        new_meta_val = meta_val
    node.meta['val'] = new_meta_val
    node.meta['tensor_meta'] = _create_meta_tensor_meta(fake_tensor_mode, new_meta_val)
    return node

def _scatter_wait_result(gm: IterGraphModule, fused_comm_block: CommBlock, comm_blocks: List[CommBlock], node_indices: Dict[fx.Node, int]) -> None:
    if False:
        i = 10
        return i + 15
    'Scatter the result of the fused communication node to the original users -- splitting the output and reshape each subitem.'
    last_wait_node_idx = 0
    for node in gm.graph.nodes:
        if node == fused_comm_block.comm_node:
            break
        last_wait_node_idx = max(node_indices.get(node, last_wait_node_idx), last_wait_node_idx)
    fused_comm_node = fused_comm_block.comm_node
    fused_wait_node = fused_comm_block.wait_nodes[0]
    with gm.graph.inserting_after(fused_wait_node):
        split_node = gm.graph.call_function(aten.split, (fused_wait_node, [int(cast(torch.Size, cb.shape).numel()) for cb in comm_blocks]))
    need_sort_nodes = []
    last_split_reshape_node = split_node
    with gm.graph.inserting_after(split_node):
        for (idx, comm_block) in enumerate(comm_blocks):
            orig_wait = comm_block.wait_nodes[0]
            nodes = collections.deque(list(orig_wait.users))
            while nodes:
                user_node = nodes.popleft()
                if not isinstance(user_node, fx.Node):
                    continue
                if node_indices[user_node] < last_wait_node_idx:
                    need_sort_nodes.append(user_node)
                    nodes.extend(list(user_node.users))
            split_idx_node = gm.graph.call_function(operator.getitem, (split_node, idx))
            with gm.graph.inserting_after(split_idx_node):
                wait_output_node = gm.graph.call_function(aten.reshape, (split_idx_node, comm_block.shape))
            gm.graph.node_replace_all_uses_with(orig_wait, wait_output_node)
        if last_split_reshape_node == split_node:
            last_split_reshape_node = wait_output_node
    need_sort_nodes = sorted(need_sort_nodes, key=lambda node: node_indices[node])
    gm.graph.move_after(need_sort_nodes, last_split_reshape_node)
    gm.graph.eliminate_dead_code()

def _fuse_with_cat(gm: IterGraphModule, comm_blocks: List[CommBlock], node_indices: Dict[fx.Node, int]) -> CommBlock:
    if False:
        return 10
    'Fuse the CommBlocks using concat given a list of CommBlock (only allreduce).'
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        if input_node.name.startswith('clone'):
            input_node = cast(fx.Node, input_node.args[0])
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            assert index != last_input_index
            last_input_node = input_node
            last_input_index = index
    with gm.graph.inserting_after(last_input_node):
        cat_inputs = []
        for input_node in all_input_nodes:
            cat_inputs.append(_call_function(gm, fake_tensor_mode, None, aten.flatten.using_ints, input_node))
    with gm.graph.inserting_after(cat_inputs[0]):
        cat_node = _call_function(gm, fake_tensor_mode, None, aten.cat, cat_inputs)
    last_comm = comm_blocks[-1]
    last_comm_node = last_comm.comm_node
    last_wait_node = last_comm.wait_nodes[0]
    with gm.graph.inserting_after(cat_node):
        (flatten_args, spec) = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = cat_node
        (args, kwargs) = tree_unflatten(flatten_args, spec)
        fused_comm_node = _call_function(gm, fake_tensor_mode, cat_node.meta['val'], last_comm_node.target, *args, **kwargs)
    with gm.graph.inserting_after(fused_comm_node):
        (flatten_args, spec) = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        (args, kwargs) = tree_unflatten(flatten_args, spec)
        fused_wait_node = _call_function(gm, fake_tensor_mode, cat_node.meta['val'], last_wait_node.target, *args, **kwargs)
    nodes_to_move = cat_inputs + [cat_node, fused_comm_node, fused_wait_node]
    gm.graph.move_after(nodes_to_move, last_input_node)
    tensor_meta = cat_node.meta.get('tensor_meta')
    fused_comm_block = CommBlock(shape=tensor_meta.shape, node_list=[fused_comm_node, fused_wait_node], wait_nodes=[fused_wait_node], comm_node=fused_comm_node, inputs=[cat_node], outputs={fused_wait_node})
    _scatter_wait_result(gm, fused_comm_block, comm_blocks, node_indices)
    return fused_comm_block

def _expedite_comm_ops(gm: IterGraphModule, comm_blocks: List[CommBlock]) -> None:
    if False:
        for i in range(10):
            print('nop')
    node_indices = {node: i for (i, node) in enumerate(gm.graph.nodes)}
    for comm_block in comm_blocks:
        last_input = comm_block.comm_node
        last_input_idx = -1
        for input in comm_block.inputs:
            input_idx = node_indices[input]
            if input_idx > last_input_idx:
                last_input = input
                last_input_idx = input_idx
        gm.graph.node_append(last_input, comm_block.comm_node)

@graph_optimization_pass(prerequisites=[], apply_after=[])
def comm_fusion_with_concat(gm: IterGraphModule, bucket_size_mb: int) -> None:
    if False:
        print('Hello World!')
    'Run fuse communication with concat.\n\n    This implementation uses concat to concat the bucketed gradients.\n    '
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, 'all_reduce'))
    _expedite_comm_ops(gm, comm_blocks)
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, 'all_reduce'))
    node_indices = {node: i for (i, node) in enumerate(gm.graph.nodes)}
    bucket_size = 1 * 1024 ** 2
    bucket_cap_size = bucket_size_mb * 1024 ** 2
    begin = end = curr_size = 0
    while end < len(comm_blocks):
        curr_size += cast(torch.Size, comm_blocks[end].shape).numel() * 4
        end += 1
        if curr_size < bucket_size:
            continue
        _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)
        bucket_size = bucket_cap_size
        begin = end
        curr_size = 0
    else:
        if begin < len(comm_blocks):
            _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)

@graph_optimization_pass(prerequisites=[comm_fusion_with_concat], apply_after=[])
def schedule_comm_wait(gm: IterGraphModule) -> None:
    if False:
        while True:
            i = 10
    'Delay the execution of wait tensors of allreduce until its first user.'
    comm_blocks = get_all_comm_blocks(gm, (CommType.ALLREDUCE, 'all_reduce'))
    allreduce_users: Set[fx.Node] = set()
    for allreduce in comm_blocks:
        for output in allreduce.outputs:
            allreduce_users.update(output.users)
    node_indices = {node: i for (i, node) in enumerate(gm.graph.nodes)}
    for allreduce in comm_blocks:
        assert len(allreduce.outputs) >= 1, f'Found a allreduce that has zero outputs/users -- {allreduce}.'
        target_node = next(iter(next(iter(allreduce.outputs)).users))
        target_node_index = 2 ** 31
        for user in (user for output in allreduce.outputs for user in output.users):
            index = node_indices[user]
            if index < target_node_index:
                target_node = user
                target_node_index = index
        wait_idx = -1
        for (wait_idx, node) in enumerate(allreduce.node_list):
            if node == allreduce.wait_nodes[0]:
                break
        assert wait_idx >= 0
        gm.graph.move_before(allreduce.node_list[wait_idx:], target_node)

@graph_optimization_pass(prerequisites=[], apply_after=[])
def remove_copy_from_optimizer(gm: IterGraphModule) -> None:
    if False:
        i = 10
        return i + 15
    'Erase the orphant copy_ that generated when tracing optimizer.\n\n    Two reasons why we could not simply use the DCE of fx.Graph.\n    1. fx.Graph treats copy_ as a side-effect node and does not erase it.\n    2. Users may want to preserve some orphan `copy_` that is not from the\n       optimizer.\n    If the second reason does not hold, this pass can be rewritten as using\n    DCE from fx.Graph (with the overwrite to the side-effect node list).\n    '
    MAX_COPY_DISTANCE = 5
    remove_candidates: Set[fx.Node] = set()
    for node in reversed(gm.graph.nodes):
        if node.users:
            continue
        if node.op != OP.CALL_FUNCTION or node.target != aten.copy_.default:
            continue
        copy_ancestors: Set[fx.Node] = set()
        nodes = collections.deque([node, None])
        distance = 0
        should_remove = False
        while nodes and distance < MAX_COPY_DISTANCE:
            visiting = nodes.popleft()
            if visiting is None:
                distance += 1
                if nodes:
                    nodes.append(None)
                continue
            copy_ancestors.add(visiting)
            if visiting.op == OP.CALL_FUNCTION and str(visiting.target).startswith(('aten._foreach_', 'aten._fused_')):
                should_remove = True
            parents = pytree.arg_tree_leaves(*visiting.args, **visiting.kwargs)
            for parent in parents:
                if isinstance(parent, fx.Node):
                    nodes.append(parent)
        if should_remove:
            remove_candidates.update(copy_ancestors)
    for node in reversed(gm.graph.nodes):
        if node.users:
            continue
        if node not in remove_candidates:
            continue
        gm.graph.erase_node(node)
AdamArgs = collections.namedtuple('AdamArgs', ['params', 'grads', 'exp_avgs', 'exp_avg_sqs', 'max_exp_avg_sqs', 'state_steps'])

@dataclass(unsafe_hash=True)
class FusedAdamBlock:
    optim_node: fx.Node
    generate_output: bool
    param_outputs: List[fx.Node] = field(default_factory=list)
    grad_outputs: List[fx.Node] = field(default_factory=list)
    exp_avgs_outputs: List[fx.Node] = field(default_factory=list)
    exp_avg_sqs_outputs: List[fx.Node] = field(default_factory=list)
    max_exp_avg_sqs: List[fx.Node] = field(default_factory=list)

    def generate_outputs(self):
        if False:
            while True:
                i = 10

        def _generate_outputs(arg_idx, output_list):
            if False:
                for i in range(10):
                    print('nop')
            graph = self.optim_node.graph
            with graph.inserting_after(self.optim_node):
                optim_getitem = graph.call_function(operator.getitem, (self.optim_node, arg_idx))
            for (i, arg) in enumerate(self.optim_node.args[arg_idx]):
                with graph.inserting_after(optim_getitem):
                    updated_arg = graph.call_function(operator.getitem, (optim_getitem, i))
                with graph.inserting_after(updated_arg):
                    output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
                output_list.append(output_copy)
        _generate_outputs(0, self.param_outputs)
        _generate_outputs(2, self.exp_avgs_outputs)
        _generate_outputs(3, self.exp_avg_sqs_outputs)

    def populate_outputs(self):
        if False:
            while True:
                i = 10

        def _populate_outputs(args_idx, output_list):
            if False:
                while True:
                    i = 10
            optim_getitem = self.optim_node
            for user in self.optim_node.users:
                assert user.target == operator.getitem, f'The user of {self.optim_node} is not getitem.'
                if user.args[1] == args_idx:
                    optim_getitem = user
                    break
            assert optim_getitem != self.optim_node, f'Cannot find the getitem node for {self.optim_node}'
            output_list.extend([self.optim_node] * len(cast(List[fx.Node], self.optim_node.args[0])))
            for updated_arg in optim_getitem.users:
                assert updated_arg.target == operator.getitem, f'Unexpected node target {updated_arg.target}.'
                idx = updated_arg.args[1]
                output_copy = next(iter(updated_arg.users))
                assert str(output_copy.target).startswith('aten.copy_'), f'Unexpected node target {output_copy.target}.'
                output_list[idx] = output_copy
            for (i, output) in enumerate(output_list):
                assert output != self.optim_node, f'{i}th output is not replaced.'
            assert output_list, f'The output for {self.optim_node} is empty.'
        _populate_outputs(0, self.param_outputs)
        _populate_outputs(2, self.exp_avgs_outputs)
        _populate_outputs(3, self.exp_avg_sqs_outputs)

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.param_outputs:
            return
        if self.generate_output:
            self.generate_outputs()
        else:
            self.populate_outputs()

@dataclass(unsafe_hash=True)
class ForeachAddBlock:
    add_node: fx.Node
    generate_output: bool
    outputs: List[fx.Node] = field(default_factory=list)

    def generate_outputs(self):
        if False:
            return 10
        graph = self.add_node.graph
        for (i, arg) in enumerate(cast(Tuple[Any, ...], self.add_node.args[0])):
            with graph.inserting_after(self.add_node):
                updated_arg = graph.call_function(operator.getitem, (self.add_node, i))
            with graph.inserting_after(updated_arg):
                output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
            self.outputs.append(output_copy)
        assert self.outputs, f'The output for {self.add_node} is empty.'

    def populate_outputs(self):
        if False:
            print('Hello World!')
        self.outputs = [self.add_node for _ in cast(Tuple[Any, ...], self.add_node.args[0])]
        for updated_arg in self.add_node.users:
            assert updated_arg.target == operator.getitem, f'Unexpected node target {updated_arg.target}'
            idx = cast(int, updated_arg.args[1])
            output_copy = next(iter(updated_arg.users))
            assert str(output_copy.target).startswith('aten.copy_'), f'The execpted output node is different, {str(output_copy.target)}'
            self.outputs[idx] = output_copy
        for (i, output) in enumerate(self.outputs):
            assert output != self.add_node, f'{i}th output is not replaced.'

    def __post_init__(self):
        if False:
            while True:
                i = 10
        if self.outputs:
            return
        if self.generate_output:
            self.generate_outputs()
        else:
            self.populate_outputs()

@dataclass(unsafe_hash=True)
class FusedOptimizerBlock:
    step: ForeachAddBlock
    optim: FusedAdamBlock

def get_fused_optimizer_block(optim_node: fx.Node) -> FusedOptimizerBlock:
    if False:
        for i in range(10):
            print('nop')
    'Given a fused optimizer node and return the FusedOptimizerBlock.'
    MAX_STEP_DISTANCE = 5
    nodes = collections.deque([optim_node, None])
    step_node = optim_node
    distance = 0
    while nodes and distance < MAX_STEP_DISTANCE:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        elif node.op == OP.CALL_FUNCTION and str(node.target).startswith('aten._foreach_add'):
            step_node = node
            break
        else:
            nodes.extend((a for a in pytree.arg_tree_leaves(*node.args, **node.kwargs) if isinstance(a, fx.Node)))
    if step_node == optim_node:
        raise RuntimeError(f'Cannot find step node (foreach_add) for the optimizer node {optim_node} with {MAX_STEP_DISTANCE} BFS distance. The API design does not match the tracing graph.')
    step = ForeachAddBlock(step_node, generate_output=False)
    optim = FusedAdamBlock(optim_node, generate_output=False)
    return FusedOptimizerBlock(step, optim)

def get_all_fused_optimizer_blocks(gm: IterGraphModule, optim_ops: Union[Tuple[str, ...], str]) -> List[FusedOptimizerBlock]:
    if False:
        while True:
            i = 10
    'Find all the FusedOptimizerBlock that the optimizer operators are in `optim_ops`.'
    return [get_fused_optimizer_block(node) for node in gm.graph.nodes if node.name.startswith(optim_ops)]

def _split_fused_adam(gm: IterGraphModule, orig_optim_block: FusedOptimizerBlock, split_gradients: Set[fx.Node]) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    if False:
        return 10
    'Split the `orig_optim_block` into two FusedOptimizerBlock.\n\n    The first one will be the optimizer that optimize `split_gradients`. The second one is\n    used to optimize the remaining gradients.\n    An assert will be raised if one of the optimizer optimize zero gradients.\n    '
    orig_optim_args = AdamArgs(*orig_optim_block.optim.optim_node.args)
    optim_args = (AdamArgs([], [], [], [], [], []), AdamArgs([], [], [], [], [], []))
    orig_optim_indices: Tuple[List[int], List[int]] = ([], [])
    orig_step_indices: Tuple[List[int], List[int]] = ([], [])
    for (idx, gradient) in enumerate(orig_optim_args.grads):
        group_idx = 0 if gradient in split_gradients else 1
        orig_optim_indices[group_idx].append(idx)
        for (orig_arg, optim_arg) in zip(orig_optim_args, optim_args[group_idx]):
            if orig_arg:
                optim_arg.append(orig_arg[idx])
        orig_step_output = optim_args[group_idx].state_steps[-1]
        assert str(orig_step_output.target).startswith('aten.copy_'), f'The copy output is {orig_step_output.target}, expect aten.copy_'
        orig_step_getitem = orig_step_output.args[1]
        assert 'getitem' in str(orig_step_getitem.target), f'The copy getitem is {orig_step_getitem.target}, expect operator.getitem'
        orig_step_idx = orig_step_getitem.args[1]
        orig_step_indices[group_idx].append(orig_step_idx)
    if not all((l for l in orig_step_indices + orig_optim_indices)):
        raise ValueError('At least one split optimizer does not have input.')
    output = get_output(gm.graph)
    results: List[FusedOptimizerBlock] = []
    (flatten_output_args, spec) = tree_flatten((output.args, output.kwargs))
    flatten_output_args_indices: DefaultDict[fx.Node, Set[int]] = collections.defaultdict(set)
    for (idx, output_arg) in enumerate(flatten_output_args):
        if isinstance(output_arg, fx.Node):
            flatten_output_args_indices[output_arg].add(idx)

    def replace_flatten_output_args(orig_node: fx.Node, new_node: fx.Node):
        if False:
            i = 10
            return i + 15
        for idx in flatten_output_args_indices[orig_node]:
            flatten_output_args[idx] = new_node
    for group_idx in range(2):
        step_args: List[fx.Node] = []
        orig_step_outputs: List[fx.Node] = []
        with gm.graph.inserting_after(orig_optim_block.optim.optim_node):
            for idx in orig_step_indices[group_idx]:
                step_args.append(cast(Tuple[fx.Node, ...], orig_optim_block.step.add_node.args[0])[idx])
                orig_step_outputs.append(orig_optim_block.step.outputs[idx])
            step = gm.graph.call_function(aten._foreach_add.Scalar, (step_args, 1))
        step_block = ForeachAddBlock(step, generate_output=True)
        for (i, step_output) in enumerate(step_block.outputs):
            orig_step_output = orig_step_outputs[i]
            replace_flatten_output_args(orig_step_output, step_output)
            assert optim_args[group_idx].state_steps[i] == orig_step_output, f'The expected step output node mismatched, {orig_step_output} {optim_args[group_idx].state_steps[i]}'
            optim_args[group_idx].state_steps[i] = step_output
        with gm.graph.inserting_after(step_block.outputs[0]):
            optim = gm.graph.call_function(aten._fused_adam.default, optim_args[group_idx], orig_optim_block.optim.optim_node.kwargs)
        optim_block = FusedAdamBlock(optim, generate_output=True)
        for (curr_idx, orig_idx) in enumerate(orig_optim_indices[group_idx]):
            list_names = ('param_outputs', 'exp_avgs_outputs', 'exp_avg_sqs_outputs')
            for name in list_names:
                orig_list = getattr(orig_optim_block.optim, name)
                curr_list = getattr(optim_block, name)
                replace_flatten_output_args(orig_list[orig_idx], curr_list[curr_idx])
        results.append(FusedOptimizerBlock(step_block, optim_block))
    (output_args, output_kwargs) = tree_unflatten(flatten_output_args, spec)
    gm.graph.node_set_args(output, output_args)
    gm.graph.node_set_kwargs(output, output_kwargs)
    for copy_output in itertools.chain(orig_optim_block.optim.param_outputs, orig_optim_block.optim.exp_avgs_outputs, orig_optim_block.optim.exp_avg_sqs_outputs):
        gm.graph.erase_node(copy_output)
    gm.graph.eliminate_dead_code()
    for copy_output in orig_optim_block.step.outputs:
        gm.graph.erase_node(copy_output)
    gm.graph.eliminate_dead_code()
    return (results[0], results[1])

def split_fused_optimizer(gm: IterGraphModule, optim_block: FusedOptimizerBlock, split_gradients: Set[fx.Node]) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    if False:
        for i in range(10):
            print('nop')
    if not split_gradients:
        raise ValueError('The given split_gradients is empty.')
    if str(optim_block.optim.optim_node.target).startswith('aten._fused_adam'):
        return _split_fused_adam(gm, optim_block, split_gradients)
    else:
        raise NotImplementedError('Only fused_adam is supported now')

@graph_optimization_pass(prerequisites=[remove_copy_from_optimizer], apply_after=[schedule_comm_wait])
def iter_move_grads_and_optimizers(gm: IterGraphModule, target_comm_node: str, target_dest_node: str) -> None:
    if False:
        print('Hello World!')
    'Extract a comm block and split out a new optimizer and step for it.\n\n    This subgraph is then moved to the forward graph.\n    '
    for comm_block in get_all_comm_blocks(gm, 'all_reduce'):
        if comm_block.comm_node.name == target_comm_node:
            break
    else:
        raise ValueError(f'Cannot find {target_comm_node}')
    optim_blocks = get_all_fused_optimizer_blocks(gm, '_fused_adam')
    for optim_block in optim_blocks:
        optim_args = AdamArgs(*optim_block.optim.optim_node.args)
        one_output = next(iter(comm_block.outputs))
        if one_output in optim_args.grads:
            break
    else:
        raise ValueError(f'{target_comm_node} is not used by any fused optimizer.')
    (move_optim, _) = split_fused_optimizer(gm, optim_block, comm_block.outputs)
    move_nodes = find_all_descendants(gm, [comm_block.comm_node, move_optim.step.add_node])
    stop_node = find_node(gm.graph, lambda n: n.name == target_dest_node)[0]
    gm.graph.move_to_next_iter_before(move_nodes, stop_node)

def find_all_descendants(gm: IterGraphModule, parent_nodes: List[fx.Node]) -> List[fx.Node]:
    if False:
        return 10
    'Identify the list of nodes to move during FX graph transformation.'
    assert len(parent_nodes) > 0, 'No parent nodes are given.'
    output = get_output(gm.graph)
    dq_parent_nodes = collections.deque(parent_nodes)
    move_node_set = set()
    while dq_parent_nodes:
        node = dq_parent_nodes.popleft()
        move_node_set.add(node)
        dq_parent_nodes += [u for u in node.users if isinstance(u, fx.Node) and u != output]
    move_nodes = [node for node in gm.graph.nodes if node in move_node_set]
    return move_nodes